import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import torch
import torch.nn.functional as F
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:  # pragma: no cover - optional dependency
    flash_attn_varlen_func = None
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel,
    Qwen3VLMoeModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)
_SKILL_LOG_ORDER = ("observing", "estimating", "scheduler", "planning")
_SKILL_LOG_INDEX = {name: idx for idx, name in enumerate(_SKILL_LOG_ORDER)}


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if flash_attn_varlen_func is None:
        raise ImportError(
            "flash_attn is required when using flattened/packed attention patching. "
            "Install flash_attn or keep data_flatten=False and data_packing=False."
        )
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    
    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    # batch, head, seq_len, dim
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # batch, seqlen, head, dim

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    attn_output = attn_output.unsqueeze(0)

    return attn_output, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils


    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = (
        return_mask
    )    
    ## qwen2_5_vl
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = (
        qwen2vl_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = (
        return_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = (
        return_mask
    )
    ## qwen3vl
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = (
        return_mask
    )
    ## qwen3vl moe
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = (
        qwen3vl_forward
    )
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = (
        return_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


class SkillAwareTrainer(Trainer):
    """Trainer with per-skill loss accounting for NavAgent multitask SFT."""

    def __init__(
        self,
        *args,
        skill_evaluator=None,
        skill_loss_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._skill_loss_totals = defaultdict(
            lambda: {"loss_sum": 0.0, "token_count": 0.0}
        )
        self._skill_log_path = Path(self.args.output_dir) / "skill_loss_history.jsonl"
        #: Optional SkillGenerationEvaluator. Set from train_qwen.py and consumed
        #: inside evaluate() below. When None, evaluate() falls back to the
        #: default HF behaviour (so we don't break users who pass an eval_dataset).
        self.skill_evaluator = skill_evaluator

        # Per-skill loss weights, normalized to mean=1. When empty, disabled.
        # Normalization keeps the aggregate loss magnitude (and hence the
        # effective LR) constant regardless of the user's weight scale; only
        # the RELATIVE contribution of each skill changes.
        self._skill_loss_weights: Dict[str, float] = {}
        if skill_loss_weights:
            normalized = {
                self._normalize_skill_name(k): float(v)
                for k, v in skill_loss_weights.items()
                if float(v) > 0
            }
            if normalized:
                mean_w = sum(normalized.values()) / len(normalized)
                self._skill_loss_weights = {
                    k: v / mean_w for k, v in normalized.items()
                }
                if self.is_world_process_zero():
                    logger.info(
                        "SkillAwareTrainer normalized skill_loss_weights "
                        f"(mean=1): {self._skill_loss_weights}"
                    )

    @staticmethod
    def _normalize_skill_name(name: Optional[str]) -> str:
        if not name:
            return "unknown"
        skill = str(name).strip().lower()
        if skill.startswith("navagent_"):
            skill = skill[len("navagent_") :]
        return skill or "unknown"

    def _accumulate_skill_losses(
        self,
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        skills: Optional[Sequence[str]],
    ) -> None:
        if logits is None or labels is None or not skills:
            return
        if logits.dim() != 3 or labels.dim() != 2:
            return
        if logits.size(0) != len(skills):
            return

        with torch.no_grad():
            # NOTE: avoid .contiguous() on the full [B, L, V] slice.
            # For Qwen3-VL with V~152k and L up to 16k that copy is ~5GB in bf16
            # and was the main driver of the DeepSpeed "allocator cache flushes"
            # warnings we saw. slicing preserves a reshape-compatible layout, so
            # flatten/reshape below work without a copy.
            shift_logits = logits[..., :-1, :].detach()
            shift_labels = labels[..., 1:].detach()

            vocab_size = shift_logits.size(-1)
            # Upcast to fp32 before cross_entropy. The HF model computes its own
            # loss in fp32 (via logits.float()), but this recompute previously ran
            # in bf16; over Qwen3-VL's ~152k vocab the bf16 log_softmax overflows
            # to inf -> nan, which is why loss_total was finite but per-skill
            # losses logged as nan. This path is under no_grad/detach so it only
            # affects logging, never the real gradients.
            token_losses = F.cross_entropy(
                shift_logits.reshape(-1, vocab_size).float(),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)
            valid_mask = shift_labels.ne(-100)

            loss_sums = torch.zeros(
                len(_SKILL_LOG_ORDER),
                device=token_losses.device,
                dtype=torch.float64,
            )
            token_counts = torch.zeros(
                len(_SKILL_LOG_ORDER),
                device=token_losses.device,
                dtype=torch.float64,
            )

            for sample_idx, skill_name in enumerate(skills):
                skill = self._normalize_skill_name(skill_name)
                skill_idx = _SKILL_LOG_INDEX.get(skill)
                if skill_idx is None:
                    continue
                sample_mask = valid_mask[sample_idx]
                if not sample_mask.any():
                    continue
                loss_sums[skill_idx] += token_losses[sample_idx][sample_mask].sum(
                    dtype=torch.float64
                )
                token_counts[skill_idx] += sample_mask.to(torch.float64).sum()

            if hasattr(self, "accelerator") and self.accelerator is not None:
                loss_sums = self.accelerator.reduce(loss_sums, reduction="sum")
                token_counts = self.accelerator.reduce(token_counts, reduction="sum")

            if self.is_world_process_zero():
                for skill, idx in _SKILL_LOG_INDEX.items():
                    count = float(token_counts[idx].item())
                    if count <= 0:
                        continue
                    self._skill_loss_totals[skill]["loss_sum"] += float(
                        loss_sums[idx].item()
                    )
                    self._skill_loss_totals[skill]["token_count"] += count

    def _should_accumulate_skill_losses(self) -> bool:
        """Only run the per-skill cross-entropy on micro-batches that belong to
        the optimizer step whose result will actually be logged. This avoids
        re-computing a full [L, V] softmax on every micro-batch (with
        gradient_accumulation_steps=8 and logging_steps=10 this drops the
        overhead by ~90%) and skips the 5GB logits materialization entirely
        on the other steps so the fused-loss path inside the HF model can run.
        """
        logging_steps = getattr(self.args, "logging_steps", 0) or 0
        if logging_steps <= 0:
            return False
        # state.global_step is incremented AFTER the optimizer step, so during
        # the micro-batches that feed step (global_step + 1) we want to match
        # "next step is a logging step".
        next_step = int(self.state.global_step) + 1
        return next_step % int(logging_steps) == 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        skills = inputs.pop("skills", None)
        accumulate = self._should_accumulate_skill_losses()
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if accumulate:
            logits = (
                outputs.get("logits")
                if isinstance(outputs, dict)
                else outputs.logits
            )
            self._accumulate_skill_losses(
                logits=logits,
                labels=inputs.get("labels"),
                skills=skills,
            )

        # Apply per-skill loss weighting. At per_device_train_batch_size=1 the
        # micro-batch has exactly one sample, so one skill key; if ever scaled
        # up we average the per-sample weights. Skills not listed in the config
        # default to 1.0 (unchanged contribution).
        if self._skill_loss_weights and skills:
            weights = [
                self._skill_loss_weights.get(
                    self._normalize_skill_name(s), 1.0
                )
                for s in skills
            ]
            scale = sum(weights) / len(weights)
            if scale != 1.0:
                loss = loss * scale

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Override HF's evaluate() to run the skill generation evaluator.

        We intentionally do NOT call super().evaluate() when a skill evaluator
        is configured, because:
        * we have no classic eval_dataset in the HF sense,
        * the HF loop would do a teacher-forced loss pass which we already
          compute during training via compute_loss, and
        * our per-skill success rate requires generation + JSON parsing that
          doesn't fit the prediction_step interface.

        When ``self.skill_evaluator`` is None we defer to the parent Trainer
        so callers using a normal eval_dataset still work.
        """
        if self.skill_evaluator is None:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        # Run the custom evaluator. It handles distributed sharding, gathers
        # params under ZeRO-3 and returns already-reduced metrics.
        metrics = self.skill_evaluator.run(trainer=self)

        # Normalize keys to <metric_key_prefix>_<...>. The evaluator already
        # produces keys starting with "eval_"; if the caller wants a different
        # prefix we honor it.
        if metric_key_prefix != "eval":
            metrics = {
                key.replace("eval_", f"{metric_key_prefix}_", 1)
                if key.startswith("eval_")
                else key: value
                for key, value in metrics.items()
            }

        # Pipe through our own log() so metrics land in trainer_state.json,
        # tensorboard (if enabled) and skill_loss_history.jsonl.
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        logs = dict(logs)
        if "loss" in logs and "loss_total" not in logs:
            logs["loss_total"] = logs["loss"]

        if self.is_world_process_zero():
            for skill in _SKILL_LOG_ORDER:
                token_count = self._skill_loss_totals[skill]["token_count"]
                if token_count <= 0:
                    continue
                logs[f"loss_{skill}"] = (
                    self._skill_loss_totals[skill]["loss_sum"] / token_count
                )

        payload = {
            "step": int(self.state.global_step),
            **logs,
        }
        if self.state.epoch is not None:
            payload["epoch"] = float(self.state.epoch)

        super().log(logs, *args, **kwargs)

        if self.is_world_process_zero():
            self._skill_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._skill_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self._skill_loss_totals = defaultdict(
            lambda: {"loss_sum": 0.0, "token_count": 0.0}
        )


# Apply monkey patches
Trainer.create_optimizer = create_optimizer

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

Qwen3VLVisionModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters