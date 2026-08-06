# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import importlib.util
import re
import torch
import transformers
import sys
from pathlib import Path
from typing import Dict

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class, SkillAwareTrainer

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.skill_eval import (
    MultiDomainSkillGenerationEvaluator,
    SkillEvalConfig,
    SkillGenerationEvaluator,
    load_train_probe_datasets,
    load_val_datasets,
)
from transformers import AutoProcessor

# ---------------------------------------------------------------------------
# torch.load 安全检查兼容 patch
# ---------------------------------------------------------------------------
# transformers>=4.56 在每次 torch.load 前调用 check_torch_load_is_safe(), 当
# torch<2.6 时直接抛 ValueError(CVE-2025-32434). 续训(resume_from_checkpoint)
# 时, HF 会用 torch.load 读 checkpoint 里的 scheduler.pt / rng_state_*.pth, 于是
# 在 torch 2.5.1 环境下被拦截而无法续训(模型权重走 safetensors 不受影响).
# 这些 .pt/.pth 都是本机训练自己产出的可信文件, 漏洞前提(加载恶意 pickle)不成立,
# 因此把该检查置为 no-op. 注意 transformers.trainer 在导入时已把该函数名绑进自身
# 命名空间, 必须 patch trainer 模块里的引用, 只 patch utils 里的无效.
import transformers.trainer as _hf_trainer

_hf_trainer.check_torch_load_is_safe = lambda *args, **kwargs: None

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _parse_eval_val_dir_spec(spec: str) -> Dict[str, Path]:
    """Parse a legacy single val dir or ``domain=/path,...`` mapping."""
    text = str(spec or "").strip()
    if not text:
        return {}
    if "=" not in text:
        return {"": Path(text)}

    parsed: Dict[str, Path] = {}
    for item in text.split(","):
        piece = item.strip()
        if not piece or "=" not in piece:
            raise ValueError(
                "eval_val_dir mappings must look like "
                "'goat=/path/to/goat,ovon=/path/to/ovon'."
            )
        domain, raw_path = (part.strip() for part in piece.split("=", 1))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", domain):
            raise ValueError(
                f"Invalid eval domain {domain!r}; use letters, digits, '_' or '-'."
            )
        if not raw_path:
            raise ValueError(f"Missing eval path for domain {domain!r}.")
        normalized_domain = domain.lower().replace("-", "_")
        if normalized_domain in parsed:
            raise ValueError(f"Duplicate eval domain: {normalized_domain!r}")
        parsed[normalized_domain] = Path(raw_path)
    return parsed


class _SkillEvalPlaceholderDataset:
    """Non-None placeholder passed to HF Trainer when the real eval path is
    the custom SkillGenerationEvaluator.

    HF Trainer.__init__ rejects ``eval_strategy != "no"`` combined with
    ``eval_dataset is None``. Our SkillAwareTrainer.evaluate() short-circuits
    to skill_evaluator.run() and never reads this object, so we just need
    something that satisfies ``is not None`` and answers ``len()`` without
    raising. A zero-length dataset is safe because HF only iterates eval_dataset
    from inside super().evaluate(), which we skip.
    """

    def __len__(self):
        return 0

    def __getitem__(self, idx):  # pragma: no cover - should never be reached
        raise IndexError(
            "SkillEvalPlaceholderDataset should never be indexed; the custom "
            "SkillGenerationEvaluator handles validation directly."
        )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)

    skill_evaluator = None
    if data_args.eval_val_dir:
        skills = tuple(
            s.strip()
            for s in (data_args.eval_skills or "").split(",")
            if s.strip()
        )
        domain_evaluators: Dict[str, SkillGenerationEvaluator] = {}
        if skills:
            eval_dirs = _parse_eval_val_dir_spec(data_args.eval_val_dir)
            for domain, val_dir in eval_dirs.items():
                val_data_by_skill = load_val_datasets(val_dir, skills=skills)
                non_empty = {k: v for k, v in val_data_by_skill.items() if v}
                train_probe_by_skill: Dict[str, list] = {}
                if (
                    non_empty
                    and getattr(data_args, "eval_train_probe_size", 0) > 0
                ):
                    # Only load train probes for skills with a non-empty val
                    # pool, and keep each domain's probe data isolated.
                    probe_skills = tuple(non_empty.keys())
                    train_probe_by_skill = load_train_probe_datasets(
                        val_dir, skills=probe_skills
                    )
                    train_probe_by_skill = {
                        k: v for k, v in train_probe_by_skill.items() if v
                    }
                if not non_empty:
                    rank0_print(
                        f"eval_val_dir={val_dir} has no requested val jsonl "
                        f"files (domain={domain or 'default'}); skipping."
                    )
                    continue

                log_name = (
                    "skill_eval_history.jsonl"
                    if not domain
                    else f"skill_eval_{domain}_history.jsonl"
                )
                domain_evaluators[domain] = SkillGenerationEvaluator(
                    processor=processor,
                    val_data_by_skill=non_empty,
                    train_probe_by_skill=train_probe_by_skill or None,
                    config=SkillEvalConfig(
                        skills=tuple(non_empty.keys()),
                        max_new_tokens=data_args.eval_max_new_tokens,
                        num_samples_per_skill=data_args.eval_num_samples_per_skill,
                        train_probe_size=getattr(
                            data_args, "eval_train_probe_size", 0
                        ),
                        log_path=Path(training_args.output_dir) / log_name,
                        shard_seed=training_args.seed,
                        planning_pixel_tolerance_px=float(
                            getattr(
                                data_args,
                                "eval_planning_pixel_tolerance",
                                64.0,
                            )
                        ),
                    ),
                )
                rank0_print(
                    "Skill generation evaluator enabled: "
                    f"domain={domain or 'default'} dir={val_dir} "
                    f"skills={list(non_empty.keys())} "
                    f"(num_samples_per_skill={data_args.eval_num_samples_per_skill}, "
                    f"train_probe_size={getattr(data_args, 'eval_train_probe_size', 0)}, "
                    f"max_new_tokens={data_args.eval_max_new_tokens}, "
                    f"planning_pix_tol={getattr(data_args, 'eval_planning_pixel_tolerance', 64.0)})"
                )

        if len(domain_evaluators) == 1 and "" in domain_evaluators:
            # Preserve legacy metric names and log path for one plain directory.
            skill_evaluator = domain_evaluators[""]
        elif domain_evaluators:
            skill_evaluator = MultiDomainSkillGenerationEvaluator(
                domain_evaluators
            )

    trainer_kwargs = dict(data_module)
    # HF Trainer.__init__ raises when eval_strategy != "no" and
    # eval_dataset is None. We bypass HF's eval loop entirely via
    # SkillAwareTrainer.evaluate() + SkillGenerationEvaluator, so the
    # eval_dataset is never read. Pass a trivial placeholder just to
    # satisfy the init-time assertion.
    if (
        skill_evaluator is not None
        and trainer_kwargs.get("eval_dataset") is None
        and str(getattr(training_args, "eval_strategy", "no")).lower() != "no"
    ):
        trainer_kwargs["eval_dataset"] = _SkillEvalPlaceholderDataset()

    # Parse per-skill loss weights spec like
    # "observing=1,estimating=2,scheduler=2,planning=2". Reuse the same
    # format parser that dataset_resample_weights uses so the UX is
    # consistent for the user.
    skill_loss_weights_dict = None
    if data_args.skill_loss_weights:
        from qwenvl.data.data_processor import parse_dataset_resample_weights

        skill_loss_weights_dict = parse_dataset_resample_weights(
            data_args.skill_loss_weights
        )

    trainer = SkillAwareTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        skill_evaluator=skill_evaluator,
        skill_loss_weights=skill_loss_weights_dict,
        **trainer_kwargs,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


def resolve_attn_implementation() -> str:
    requested = os.environ.get(
        "QWENVL_ATTN_IMPLEMENTATION", "flash_attention_2"
    )
    if (
        requested == "flash_attention_2"
        and importlib.util.find_spec("flash_attn") is None
    ):
        logging.warning(
            "flash_attn is unavailable; falling back to sdpa attention."
        )
        return "sdpa"
    return requested


if __name__ == "__main__":
    train(attn_implementation=resolve_attn_implementation())
