"""Generation-based per-skill evaluation for NavAgent SFT.

We sample held-out episodes (see ``tools/split_navagent_val.py``) and, at
configurable training steps, run ``model.generate`` on each val sample, parse
the JSON response and compare the key field against the oracle:

  estimating : status              in {Normal, Lost, Completed}
                                   → correctness = status matches oracle.
  planning   : selected_direction  in {front, right, left, back}
                                   + target_pixel = [u, v]
                                   → we report THREE accuracies:
                                       dir_acc   - direction correct
                                       pix_acc   - target_pixel within
                                                   ``pixel_tolerance_px`` L2
                                                   of oracle pixel
                                       joint_acc - both above are true
                                     The headline ``acc`` is joint_acc; the
                                     other two are surfaced as extra metrics
                                     for diagnosis.

For planning we also break down dir/pix/joint by oracle direction so the user
can see e.g. "front pixel accuracy" separately from "back pixel accuracy" —
similar to per-class status recall on estimating.

The evaluator is distributed-aware: val samples are sharded across ranks, each
rank generates for its shard, and counts are ``all_reduce``-summed at the end.
Under DeepSpeed ZeRO-3 we wrap the generate loop in
``deepspeed.zero.GatheredParameters`` so that every layer's parameters are
materialized once for the whole eval pass instead of being all-gathered per
decode step (which would make generation prohibitively slow).
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from qwenvl.data.data_processor import _build_messages, _normalize_skill_name


def _is_main_process() -> bool:
    """True if this is the main process (rank 0 in distributed, or single
    process otherwise). Used to gate stdout so eval output isn't duplicated
    across all ranks of the torchrun launcher."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ---------------------------------------------------------------------------
# Per-skill scoring
# ---------------------------------------------------------------------------

#: Maps skill name -> (oracle_field_name, set_of_valid_values_or_None).
#: set_of_valid_values is used to defensively normalize predictions (lowercase
#: and membership check) for planning; for estimating the canonical values are
#: capitalized so we fall back to case-insensitive comparison.
_SKILL_FIELD_SPEC: Dict[str, Tuple[str, Optional[set]]] = {
    "estimating": ("status", {"Normal", "Lost", "Completed"}),
    "planning": ("selected_direction", {"front", "right", "left", "back"}),
}

#: Display order for per-class breakdown (kept stable across eval points).
_SKILL_LABEL_ORDER: Dict[str, Tuple[str, ...]] = {
    "estimating": ("Normal", "Lost", "Completed"),
    "planning": ("front", "right", "left", "back"),
}

#: Display order for per-goal-type breakdown.  The val set is split 1:1:1
#: across these by ``tools/split_navagent_val.py --sub-stratify-skill``.
_GOAL_TYPE_ORDER: Tuple[str, ...] = ("object", "description", "image")


def _canonicalize_label(skill: str, value: Any) -> Optional[str]:
    """Normalize a predicted/oracle label into the canonical casing defined
    by ``_SKILL_LABEL_ORDER``. Returns ``None`` for unknown / missing labels
    (callers bucket them as ``"other"``)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    for canon in _SKILL_LABEL_ORDER.get(skill, ()):
        if canon.lower() == text:
            return canon
    return None


def _extract_json_blob(text: str) -> Optional[dict]:
    """Best-effort JSON extractor. The training prompt asks for bare JSON with
    no markdown fences, but models occasionally wrap it in ```json``` anyway
    during the early training stages, so we strip a single fenced block if
    present before trying to parse.
    """
    if not text:
        return None
    stripped = text.strip()
    # Strip markdown fence if present.
    fence_match = re.match(
        r"^```(?:json)?\s*(?P<body>.*?)\s*```$", stripped, flags=re.DOTALL
    )
    if fence_match:
        stripped = fence_match.group("body").strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Second chance: grab the first balanced {...} substring.
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(stripped[start : idx + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
                break
    return None


def _compare_field(skill: str, predicted: Any, oracle: Any) -> bool:
    if predicted is None or oracle is None:
        return False
    if skill == "planning":
        return str(predicted).strip().lower() == str(oracle).strip().lower()
    if skill == "estimating":
        return str(predicted).strip().lower() == str(oracle).strip().lower()
    return str(predicted) == str(oracle)


def _coerce_pixel(value: Any) -> Optional[Tuple[float, float]]:
    """Best-effort convert a target_pixel field to a (u, v) float tuple.

    Accepts ``[u, v]`` lists/tuples of length 2 whose elements are numeric
    (int / float / numeric string). Returns ``None`` for any other shape so
    callers can treat it as a parse failure.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    out: List[float] = []
    for v in value:
        if isinstance(v, bool):  # bool is a subtype of int, exclude it
            return None
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            try:
                out.append(float(v.strip()))
            except ValueError:
                return None
        else:
            return None
    return out[0], out[1]


def _pixel_within_tol_l2(pred: Any, oracle: Any, tol_px: float) -> bool:
    """Return True iff both pixels parse and ||pred - oracle||_2 < tol_px.

    L2 (Euclidean) distance with strict ``<`` matches the user-facing
    "within X pixels" framing. Parse failures on either side are counted as
    incorrect; the caller separately tracks parse rate.
    """
    if tol_px <= 0:
        return False
    p = _coerce_pixel(pred)
    o = _coerce_pixel(oracle)
    if p is None or o is None:
        return False
    du = p[0] - o[0]
    dv = p[1] - o[1]
    return math.hypot(du, dv) < float(tol_px)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class SkillEvalConfig:
    skills: Tuple[str, ...] = ("estimating", "planning")
    max_new_tokens: int = 512
    num_samples_per_skill: int = 300
    # When > 0, also probe the TRAIN split each eval cycle. Use a small fixed
    # subset (e.g. 96) sampled deterministically from the train file. The
    # gap between train_acc and val_acc tells us whether learning is happening
    # at all — if train_acc also stays flat, the model isn't fitting the data
    # (data quality / lr / capacity issue), not just generalization failure.
    train_probe_size: int = 0
    log_path: Optional[Path] = None
    shard_seed: int = 0
    # If True, before each eval pass we subsample val to num_samples_per_skill
    # deterministically (based on shard_seed + eval_id). If False, we always
    # evaluate the full val split.
    subsample: bool = True
    # L2 pixel tolerance for planning's target_pixel: predicted pixel counts
    # as "pixel correct" iff the L2 (Euclidean) distance to the oracle pixel
    # is strictly less than this many pixels. Default 64 ≈ 10% of the 640px
    # image width — generous enough that small-scale fixation jitter doesn't
    # show up as a regression, tight enough that "model points at the wrong
    # room" is still caught. Override per-run via --eval_pixel_tolerance.
    planning_pixel_tolerance_px: float = 64.0


@dataclass
class _SkillResult:
    total: int = 0
    parseable: int = 0
    correct: int = 0
    # Per-class confusion. Keys come from ``_SKILL_LABEL_ORDER`` (or
    # ``"other"`` / ``"parse_fail"`` buckets for predictions). We track:
    #   class_total[l]   = #samples whose ORACLE label is l
    #   class_correct[l] = #samples whose ORACLE label is l AND pred matches
    #                      headline criterion (status for estimating, joint
    #                      direction+pixel for planning)
    #   pred_total[l]    = #samples whose PRED is l (regardless of oracle).
    # The (class_total, pred_total) pair lets us spot "model just always
    # says Normal" pathologies even when overall acc looks reasonable.
    class_total: Dict[str, int] = field(default_factory=dict)
    class_correct: Dict[str, int] = field(default_factory=dict)
    pred_total: Dict[str, int] = field(default_factory=dict)

    # Planning-specific decomposition. For estimating these stay empty.
    # ``dir_correct`` counts samples whose predicted direction matches the
    # oracle direction (regardless of pixel). ``pix_correct`` counts samples
    # whose predicted pixel is within the configured L2 tolerance of the
    # oracle pixel (regardless of direction). Per-class versions are
    # bucketed by ORACLE direction so the user can see e.g. "back pixel
    # acc" separately, just like Normal/Completed recall on estimating.
    dir_correct: int = 0
    pix_correct: int = 0
    class_dir_correct: Dict[str, int] = field(default_factory=dict)
    class_pix_correct: Dict[str, int] = field(default_factory=dict)

    # Per-goal-type breakdown.  ``goal_type_total[g]`` = #samples whose
    # ``meta.goal_type`` is ``g``; ``goal_type_correct[g]`` = subset of
    # those that the model got right under the skill's headline criterion
    # (status for estimating, joint dir+pix for planning).  Planning also
    # carries dir-only / pix-only goal-type counters.
    goal_type_total: Dict[str, int] = field(default_factory=dict)
    goal_type_correct: Dict[str, int] = field(default_factory=dict)
    goal_type_dir_correct: Dict[str, int] = field(default_factory=dict)
    goal_type_pix_correct: Dict[str, int] = field(default_factory=dict)

    # Full (oracle_label × goal_type) matrix breakdown.  Tuple keys make
    # serialisation via all_gather_object straightforward.  For estimating
    # this yields 2×3 = 6 buckets aligned with the val split; for planning
    # it's 4×3 = 12 buckets, also aligned with the val split.
    matrix_total: Dict[Tuple[str, str], int] = field(default_factory=dict)
    matrix_correct: Dict[Tuple[str, str], int] = field(default_factory=dict)
    matrix_dir_correct: Dict[Tuple[str, str], int] = field(default_factory=dict)
    matrix_pix_correct: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def as_dict(self, prefix: str) -> Dict[str, float]:
        total = max(self.total, 1)
        out: Dict[str, float] = {
            f"{prefix}_acc": self.correct / total,
            f"{prefix}_parse_rate": self.parseable / total,
            f"{prefix}_total": float(self.total),
        }
        # Headline auxiliary metrics for planning. For estimating these are
        # zero (we don't bump dir_correct/pix_correct there) which is fine —
        # they just don't surface in the log line below either.
        if self.dir_correct or self.pix_correct:
            out[f"{prefix}_dir_acc"] = self.dir_correct / total
            out[f"{prefix}_pix_acc"] = self.pix_correct / total
        # Surface per-class recall in trainer metrics so we can later plot
        # Normal-acc vs Completed-acc curves separately, not just the mean.
        for label, n in self.class_total.items():
            if n <= 0:
                continue
            joint = self.class_correct.get(label, 0)
            out[f"{prefix}_{label.lower()}_recall"] = joint / n
            out[f"{prefix}_{label.lower()}_n"] = float(n)
            # Per-class direction / pixel breakdown for planning. Skipped
            # when both bump-counters are zero (i.e. estimating).
            d = self.class_dir_correct.get(label, 0)
            p = self.class_pix_correct.get(label, 0)
            if d or p:
                out[f"{prefix}_{label.lower()}_dir_recall"] = d / n
                out[f"{prefix}_{label.lower()}_pix_recall"] = p / n
        # Per-goal-type accuracy (object / description / image).  Lets
        # the user watch e.g. description SR curve separately from object.
        for gt, n in self.goal_type_total.items():
            if n <= 0:
                continue
            out[f"{prefix}_gt_{gt}_acc"] = self.goal_type_correct.get(gt, 0) / n
            out[f"{prefix}_gt_{gt}_n"] = float(n)
            d = self.goal_type_dir_correct.get(gt, 0)
            p = self.goal_type_pix_correct.get(gt, 0)
            if d or p:
                out[f"{prefix}_gt_{gt}_dir_acc"] = d / n
                out[f"{prefix}_gt_{gt}_pix_acc"] = p / n
        # Full (label × goal_type) matrix.  Stable metric names use
        # ``{label}_{goal_type}`` (lowercased label) so e.g.
        # ``val_estimating_completed_description_recall`` is plottable.
        for (label, gt), n in self.matrix_total.items():
            if n <= 0:
                continue
            joint = self.matrix_correct.get((label, gt), 0)
            out[f"{prefix}_{label.lower()}_{gt}_recall"] = joint / n
            out[f"{prefix}_{label.lower()}_{gt}_n"] = float(n)
            d = self.matrix_dir_correct.get((label, gt), 0)
            p = self.matrix_pix_correct.get((label, gt), 0)
            if d or p:
                out[f"{prefix}_{label.lower()}_{gt}_dir_recall"] = d / n
                out[f"{prefix}_{label.lower()}_{gt}_pix_recall"] = p / n
        return out


class SkillGenerationEvaluator:
    """Runs periodic generate+JSON-compare evaluation during training.

    Usage (from the Trainer):

        evaluator = SkillGenerationEvaluator(
            processor=processor,
            val_data_by_skill={"estimating": [...], "planning": [...]},
            config=SkillEvalConfig(...),
        )
        metrics = evaluator.run(trainer=self)
        self.log(metrics)
    """

    def __init__(
        self,
        processor,
        val_data_by_skill: Dict[str, List[dict]],
        config: SkillEvalConfig,
        train_probe_by_skill: Optional[Dict[str, List[dict]]] = None,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.val_data_by_skill = val_data_by_skill
        # Optional pool sampled from the train split (NOT the same shards
        # as the trainer sees per step — we just open the train jsonl and
        # take a small fixed subset for monitoring).
        self.train_probe_by_skill = train_probe_by_skill or {}
        self.config = config
        self._eval_counter = 0

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self, trainer) -> Dict[str, float]:
        model = trainer.model
        accelerator = getattr(trainer, "accelerator", None)
        rank = self._rank(accelerator)
        world_size = self._world_size(accelerator)
        device = self._device(model, accelerator)
        is_dist = dist.is_available() and dist.is_initialized()

        eval_id = self._eval_counter
        self._eval_counter += 1

        # Sync everyone before eval starts so that no rank is still inside
        # the previous training step's collective when we toggle eval mode.
        if is_dist:
            dist.barrier()

        was_training = model.training
        model.eval()
        saved_use_cache = getattr(model.config, "use_cache", False)
        model.config.use_cache = True

        metrics: Dict[str, float] = {}
        start_time = time.time()

        # Data-parallel eval: every rank materializes the full model and
        # generates its own shard of samples. modifier_rank=None tells
        # DeepSpeed all ranks need the materialized params for forward.
        # We pay one all-gather at context entry, then per-token generate
        # is local on each rank (no per-step all-gather).
        # The historical "8-way deadlock" came from variable-length output
        # racing collective ops; we defuse it with synced_gpus=True in
        # generate(), so every rank participates in every decode step
        # until all reach EOS or max_new_tokens.
        gather_ctx = self._build_gather_context(model, target_rank=None)

        # List of (split_tag, pool_dict, target_size, metric_prefix). Order
        # matters — val is what the user cares about, train probe is the
        # secondary sanity signal, so we do val first.
        splits: List[Tuple[str, Dict[str, List[dict]], int, str]] = [
            (
                "val",
                self.val_data_by_skill,
                self.config.num_samples_per_skill,
                "eval",
            ),
        ]
        if self.train_probe_by_skill and self.config.train_probe_size > 0:
            splits.append(
                (
                    "train",
                    self.train_probe_by_skill,
                    self.config.train_probe_size,
                    "train",
                )
            )

        try:
            with torch.no_grad(), gather_ctx:
                for skill in self.config.skills:
                    for split_tag, pool_dict, target_size, prefix_kind in splits:
                        pool = pool_dict.get(skill) or []
                        if not pool:
                            continue
                        samples = self._pick_subset(pool, target_size)
                        if not samples:
                            continue
                        # Round-robin shard so every rank gets a similar
                        # workload mix (some samples may be longer than
                        # others; contiguous shards would imbalance).
                        if world_size > 1:
                            my_samples = samples[rank::world_size]
                        else:
                            my_samples = samples
                        skill_start = time.time()
                        result = self._evaluate_skill(
                            skill, model, my_samples, device
                        )
                        # Aggregate per-rank counts into a global tally.
                        self._all_reduce_result(result, device)
                        prefix = f"{prefix_kind}_{skill}"
                        metrics.update(result.as_dict(prefix))
                        dt = time.time() - skill_start
                        if rank == 0:
                            tag = f"{skill} ({split_tag})"
                            total = max(result.total, 1)
                            if skill == "planning":
                                print(
                                    f"[skill_eval] {tag}: "
                                    f"joint={result.correct}/{result.total}"
                                    f" ({result.correct / total * 100:.1f}%)"
                                    f"  dir={result.dir_correct}/{result.total}"
                                    f" ({result.dir_correct / total * 100:.1f}%)"
                                    f"  pix={result.pix_correct}/{result.total}"
                                    f" ({result.pix_correct / total * 100:.1f}%)"
                                    f"  parse_ok={result.parseable}/{result.total}"
                                    f" ({dt:.1f}s, {world_size}-way DP)",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[skill_eval] {tag}: "
                                    f"acc={result.correct}/{result.total}"
                                    f" ({result.correct / total * 100:.1f}%)"
                                    f" parse_ok={result.parseable}/{result.total}"
                                    f" ({dt:.1f}s, {world_size}-way DP)",
                                    flush=True,
                                )
                            # Per-class recall: catches "model just always
                            # says <majority>" pathologies that overall acc
                            # hides. For estimating this is e.g.
                            # Normal=92/96 (96%) Completed=10/96 (10%); for
                            # planning we also report dir-only and pix-only
                            # recall on a second line so you can tell
                            # "front pixel acc" from "front dir acc".
                            labels = _SKILL_LABEL_ORDER.get(skill, ())
                            joint_parts: List[str] = []
                            for label in labels:
                                n = result.class_total.get(label, 0)
                                ok = result.class_correct.get(label, 0)
                                if n > 0:
                                    joint_parts.append(
                                        f"{label}={ok}/{n} ({ok / n * 100:.0f}%)"
                                    )
                                else:
                                    joint_parts.append(f"{label}=0/0")
                            metric_word = "joint" if skill == "planning" else "recall"
                            print(
                                f"[skill_eval] {tag}   per-class {metric_word}: "
                                + "  ".join(joint_parts),
                                flush=True,
                            )
                            if skill == "planning":
                                dir_parts: List[str] = []
                                pix_parts: List[str] = []
                                for label in labels:
                                    n = result.class_total.get(label, 0)
                                    d = result.class_dir_correct.get(label, 0)
                                    p = result.class_pix_correct.get(label, 0)
                                    if n > 0:
                                        dir_parts.append(
                                            f"{label}={d}/{n} ({d / n * 100:.0f}%)"
                                        )
                                        pix_parts.append(
                                            f"{label}={p}/{n} ({p / n * 100:.0f}%)"
                                        )
                                    else:
                                        dir_parts.append(f"{label}=0/0")
                                        pix_parts.append(f"{label}=0/0")
                                print(
                                    f"[skill_eval] {tag}   per-class dir:   "
                                    + "  ".join(dir_parts),
                                    flush=True,
                                )
                                print(
                                    f"[skill_eval] {tag}   per-class pix:   "
                                    + "  ".join(pix_parts),
                                    flush=True,
                                )
                            # Predicted-label distribution. If e.g. Normal=95
                            # while train shows Completed should be ~17%, the
                            # model has collapsed onto the majority class.
                            pred_parts: List[str] = []
                            for label in labels:
                                cnt = result.pred_total.get(label, 0)
                                if cnt > 0:
                                    pred_parts.append(f"{label}={cnt}")
                            for extra in ("other", "parse_fail"):
                                cnt = result.pred_total.get(extra, 0)
                                if cnt > 0:
                                    pred_parts.append(f"{extra}={cnt}")
                            print(
                                f"[skill_eval] {tag}   pred-dist: "
                                + "  ".join(pred_parts),
                                flush=True,
                            )
                            # Per-goal-type breakdown.  Lets the user see
                            # at a glance whether description / image /
                            # object are being learned at different paces.
                            gt_parts: List[str] = []
                            for gt in _GOAL_TYPE_ORDER:
                                n = result.goal_type_total.get(gt, 0)
                                ok = result.goal_type_correct.get(gt, 0)
                                if n > 0:
                                    gt_parts.append(
                                        f"{gt}={ok}/{n} ({ok / n * 100:.0f}%)"
                                    )
                                else:
                                    gt_parts.append(f"{gt}=0/0")
                            print(
                                f"[skill_eval] {tag}   per-goal-type "
                                f"{metric_word}: " + "  ".join(gt_parts),
                                flush=True,
                            )
                            if skill == "planning":
                                gt_dir_parts: List[str] = []
                                gt_pix_parts: List[str] = []
                                for gt in _GOAL_TYPE_ORDER:
                                    n = result.goal_type_total.get(gt, 0)
                                    d = result.goal_type_dir_correct.get(gt, 0)
                                    p = result.goal_type_pix_correct.get(gt, 0)
                                    if n > 0:
                                        gt_dir_parts.append(
                                            f"{gt}={d}/{n} ({d / n * 100:.0f}%)"
                                        )
                                        gt_pix_parts.append(
                                            f"{gt}={p}/{n} ({p / n * 100:.0f}%)"
                                        )
                                    else:
                                        gt_dir_parts.append(f"{gt}=0/0")
                                        gt_pix_parts.append(f"{gt}=0/0")
                                print(
                                    f"[skill_eval] {tag}   per-goal-type dir: "
                                    + "  ".join(gt_dir_parts),
                                    flush=True,
                                )
                                print(
                                    f"[skill_eval] {tag}   per-goal-type pix: "
                                    + "  ".join(gt_pix_parts),
                                    flush=True,
                                )
                            # Full (label × goal_type) matrix.  One line
                            # per primary label, columns aligned across
                            # goal_types so it reads like a small confusion
                            # matrix.
                            print(
                                f"[skill_eval] {tag}   per-class x goal_type "
                                f"{metric_word}:",
                                flush=True,
                            )
                            for label in labels:
                                cells: List[str] = []
                                for gt in _GOAL_TYPE_ORDER:
                                    n = result.matrix_total.get((label, gt), 0)
                                    ok = result.matrix_correct.get(
                                        (label, gt), 0
                                    )
                                    if n > 0:
                                        cells.append(
                                            f"{gt}={ok}/{n} ({ok / n * 100:.0f}%)"
                                        )
                                    else:
                                        cells.append(f"{gt}=0/0")
                                print(
                                    f"[skill_eval]              {label:11s}  "
                                    + "  ".join(cells),
                                    flush=True,
                                )
        finally:
            model.config.use_cache = saved_use_cache
            if was_training:
                model.train()

        metrics["eval_runtime"] = time.time() - start_time

        # Broadcast metrics from rank 0 so HF Trainer's log()/callbacks see
        # consistent values on every rank (otherwise on_evaluate callbacks
        # could disagree on whether eval ran).
        metrics = self._broadcast_metrics_dict(metrics, src=0)

        # Final barrier so that training resumes in lockstep on all ranks.
        if is_dist:
            dist.barrier()

        if rank == 0 and self.config.log_path is not None:
            self._append_log(metrics, trainer)
        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pick_subset(self, pool: List[dict], target_size: int) -> List[dict]:
        """Pick a deterministic, fixed subset from ``pool``. Used for both
        val and the optional train probe. Pinned to ``shard_seed`` so the
        subset is reproducible across eval checkpoints — the only thing
        changing across eval points is the model, not the samples (a
        per-eval reshuffle would otherwise add ~10pp of sample-difficulty
        noise that swamps real per-step signal)."""
        if not pool:
            return []
        if not self.config.subsample:
            return list(pool)
        target = min(target_size, len(pool))
        if target <= 0:
            return []
        if target == len(pool):
            return list(pool)
        gen = torch.Generator()
        gen.manual_seed(self.config.shard_seed)
        perm = torch.randperm(len(pool), generator=gen).tolist()
        return [pool[i] for i in perm[:target]]

    def _evaluate_skill(
        self,
        skill: str,
        model,
        samples: List[dict],
        device: torch.device,
    ) -> _SkillResult:
        field, _valid = _SKILL_FIELD_SPEC.get(skill, (None, None))
        if field is None:
            return _SkillResult()
        result = _SkillResult()
        # Force a newline so we don't share a line with the training tqdm bar
        # (HF Trainer leaves the training bar without a trailing newline when
        # eval kicks in, which makes our \r-based bar invisible). Also write
        # to stderr explicitly and bump mininterval/miniters so updates flush
        # even when stdout is captured by torchrun.
        if _is_main_process():
            sys.stderr.write("\n")
            sys.stderr.flush()
        pbar = tqdm(
            samples,
            desc=f"[skill_eval] {skill}",
            total=len(samples),
            disable=not _is_main_process(),
            dynamic_ncols=True,
            leave=True,
            file=sys.stderr,
            mininterval=0.5,
            miniters=1,
        )
        tol_px = float(self.config.planning_pixel_tolerance_px)
        for sample in pbar:
            try:
                oracle_text = sample["conversations"][-1]["value"]
                oracle = _extract_json_blob(oracle_text) or {}
                oracle_canon = _canonicalize_label(skill, oracle.get(field))
                # Bucket by oracle's canonical label for per-class denominator.
                # Unknown / missing oracle labels go into "other"; this should
                # be rare (the val data is teacher-validated) but we keep them
                # so totals always reconcile.
                oracle_key = oracle_canon or "other"
                result.class_total[oracle_key] = (
                    result.class_total.get(oracle_key, 0) + 1
                )

                # Goal-type bucket (object/description/image).  Samples
                # without a recognised goal_type fall into "other" so
                # totals reconcile.
                gt_raw = str(
                    sample.get("meta", {}).get("goal_type", "") or ""
                ).strip().lower()
                gt_key = gt_raw if gt_raw in _GOAL_TYPE_ORDER else "other"
                result.goal_type_total[gt_key] = (
                    result.goal_type_total.get(gt_key, 0) + 1
                )
                if oracle_canon is not None:
                    matrix_key = (oracle_canon, gt_key)
                    result.matrix_total[matrix_key] = (
                        result.matrix_total.get(matrix_key, 0) + 1
                    )

                predicted_json = self._generate_for_sample(sample, model, device)
                result.total += 1
                if predicted_json is None:
                    # Parse failure: count against pred distribution but not
                    # against any class_correct bucket.
                    result.pred_total["parse_fail"] = (
                        result.pred_total.get("parse_fail", 0) + 1
                    )
                    continue
                result.parseable += 1

                pred_canon = _canonicalize_label(skill, predicted_json.get(field))
                pred_key = pred_canon or "other"
                result.pred_total[pred_key] = (
                    result.pred_total.get(pred_key, 0) + 1
                )

                dir_ok = _compare_field(skill, predicted_json.get(field), oracle.get(field))
                if skill == "planning":
                    pix_ok = _pixel_within_tol_l2(
                        predicted_json.get("target_pixel"),
                        oracle.get("target_pixel"),
                        tol_px,
                    )
                    # Track the two single-axis accuracies independently…
                    if dir_ok:
                        result.dir_correct += 1
                        result.goal_type_dir_correct[gt_key] = (
                            result.goal_type_dir_correct.get(gt_key, 0) + 1
                        )
                        if oracle_canon is not None:
                            result.class_dir_correct[oracle_canon] = (
                                result.class_dir_correct.get(oracle_canon, 0) + 1
                            )
                            result.matrix_dir_correct[(oracle_canon, gt_key)] = (
                                result.matrix_dir_correct.get(
                                    (oracle_canon, gt_key), 0
                                ) + 1
                            )
                    if pix_ok:
                        result.pix_correct += 1
                        result.goal_type_pix_correct[gt_key] = (
                            result.goal_type_pix_correct.get(gt_key, 0) + 1
                        )
                        if oracle_canon is not None:
                            result.class_pix_correct[oracle_canon] = (
                                result.class_pix_correct.get(oracle_canon, 0) + 1
                            )
                            result.matrix_pix_correct[(oracle_canon, gt_key)] = (
                                result.matrix_pix_correct.get(
                                    (oracle_canon, gt_key), 0
                                ) + 1
                            )
                    # …and use joint (both correct) as the headline metric.
                    if dir_ok and pix_ok:
                        result.correct += 1
                        result.goal_type_correct[gt_key] = (
                            result.goal_type_correct.get(gt_key, 0) + 1
                        )
                        if oracle_canon is not None:
                            result.class_correct[oracle_canon] = (
                                result.class_correct.get(oracle_canon, 0) + 1
                            )
                            result.matrix_correct[(oracle_canon, gt_key)] = (
                                result.matrix_correct.get(
                                    (oracle_canon, gt_key), 0
                                ) + 1
                            )
                else:
                    # estimating (and anything else with no pixel side).
                    if dir_ok:
                        result.correct += 1
                        result.goal_type_correct[gt_key] = (
                            result.goal_type_correct.get(gt_key, 0) + 1
                        )
                        if oracle_canon is not None:
                            result.class_correct[oracle_canon] = (
                                result.class_correct.get(oracle_canon, 0) + 1
                            )
                            result.matrix_correct[(oracle_canon, gt_key)] = (
                                result.matrix_correct.get(
                                    (oracle_canon, gt_key), 0
                                ) + 1
                            )
            except Exception as exc:  # pragma: no cover - defensive
                # We don't want a single bad sample to break the whole eval.
                # Only rank 0 actually runs generate() in the new path, but
                # guard anyway in case this is called outside that scope.
                if _is_main_process():
                    print(
                        f"[skill_eval] WARNING: sample failed ({skill}): {exc}",
                        flush=True,
                    )
                result.total += 1
            denom = max(result.total, 1)
            if skill == "planning":
                pbar.set_postfix(
                    joint=f"{result.correct}/{result.total}",
                    dir=f"{result.dir_correct}/{result.total}",
                    pix=f"{result.pix_correct}/{result.total}",
                    pct=f"{(result.correct / denom) * 100:.1f}%",
                )
            else:
                pbar.set_postfix(
                    acc=f"{result.correct}/{result.total}",
                    pct=f"{(result.correct / denom) * 100:.1f}%",
                    parse=f"{result.parseable}/{result.total}",
                )
        pbar.close()
        return result

    def _generate_for_sample(
        self, sample: dict, model, device: torch.device
    ) -> Optional[dict]:
        base_path = Path(sample.get("data_path", ""))
        # Drop the oracle (assistant) turn; _build_messages expects placeholders
        # to be consumed only on the user side, so this is safe.
        prompt_sample = dict(sample)
        prompt_sample["conversations"] = [
            turn
            for turn in sample["conversations"]
            if turn.get("from") == "human"
        ]
        if not prompt_sample["conversations"]:
            return None
        messages = _build_messages(prompt_sample, base_path)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        )
        if self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        # NOTE: We deliberately do NOT set synced_gpus=True. Because the
        # outer GatheredParameters(modifier_rank=None) context has already
        # materialized the full model on every rank, per-decode-step
        # forward is purely local compute — no cross-rank collective is
        # needed. Setting synced_gpus=True in that case forces empty
        # NCCL barriers on every token (~5x slowdown) without any
        # correctness benefit. The "different ranks finish at different
        # times" deadlock only happens when params are sharded, which
        # is not our case here.
        output = model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _extract_json_blob(text)

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _world_size(accelerator) -> int:
        if accelerator is not None and hasattr(accelerator, "num_processes"):
            return int(accelerator.num_processes)
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _rank(accelerator) -> int:
        if accelerator is not None and hasattr(accelerator, "process_index"):
            return int(accelerator.process_index)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _device(model, accelerator) -> torch.device:
        if accelerator is not None and hasattr(accelerator, "device"):
            return accelerator.device
        if hasattr(model, "device"):
            return model.device
        for p in model.parameters():
            return p.device
        return torch.device("cpu")

    def _all_reduce_result(self, result: _SkillResult, device: torch.device) -> None:
        # Reduce scalar counters with a tensor all-reduce (cheap), and the
        # per-class dicts via all_gather_object (small payload, but with
        # variable keys so a tensor encoding would be brittle).
        if not (dist.is_available() and dist.is_initialized()):
            return
        tensor = torch.tensor(
            [
                result.correct,
                result.parseable,
                result.total,
                result.dir_correct,
                result.pix_correct,
            ],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        result.correct = int(tensor[0].item())
        result.parseable = int(tensor[1].item())
        result.total = int(tensor[2].item())
        result.dir_correct = int(tensor[3].item())
        result.pix_correct = int(tensor[4].item())

        world_size = dist.get_world_size()
        payload = {
            "class_total": dict(result.class_total),
            "class_correct": dict(result.class_correct),
            "pred_total": dict(result.pred_total),
            "class_dir_correct": dict(result.class_dir_correct),
            "class_pix_correct": dict(result.class_pix_correct),
            "goal_type_total": dict(result.goal_type_total),
            "goal_type_correct": dict(result.goal_type_correct),
            "goal_type_dir_correct": dict(result.goal_type_dir_correct),
            "goal_type_pix_correct": dict(result.goal_type_pix_correct),
            "matrix_total": dict(result.matrix_total),
            "matrix_correct": dict(result.matrix_correct),
            "matrix_dir_correct": dict(result.matrix_dir_correct),
            "matrix_pix_correct": dict(result.matrix_pix_correct),
        }
        gathered: List[Optional[dict]] = [None] * world_size
        dist.all_gather_object(gathered, payload)

        merged: Dict[str, Dict[Any, int]] = {k: {} for k in payload}
        for shard in gathered:
            if not shard:
                continue
            for bucket_name in merged.keys():
                for k, v in (shard.get(bucket_name) or {}).items():
                    merged[bucket_name][k] = merged[bucket_name].get(k, 0) + int(v)
        result.class_total = merged["class_total"]
        result.class_correct = merged["class_correct"]
        result.pred_total = merged["pred_total"]
        result.class_dir_correct = merged["class_dir_correct"]
        result.class_pix_correct = merged["class_pix_correct"]
        result.goal_type_total = merged["goal_type_total"]
        result.goal_type_correct = merged["goal_type_correct"]
        result.goal_type_dir_correct = merged["goal_type_dir_correct"]
        result.goal_type_pix_correct = merged["goal_type_pix_correct"]
        result.matrix_total = merged["matrix_total"]
        result.matrix_correct = merged["matrix_correct"]
        result.matrix_dir_correct = merged["matrix_dir_correct"]
        result.matrix_pix_correct = merged["matrix_pix_correct"]

    # ------------------------------------------------------------------
    # ZeRO-3 parameter gathering
    # ------------------------------------------------------------------

    def _build_gather_context(self, model, target_rank: Optional[int] = None):
        """Returns a context manager that materializes all parameters for the
        duration of generate().

        ``target_rank=None`` (default, used for data-parallel eval): every
        rank materializes the full model, so each rank can run its own
        ``model.generate`` shard without per-token all-gathers.

        ``target_rank=int``: only the named rank holds the materialized
        params; other ranks just participate in the collective at enter/exit
        and must NOT run forward inside the context. This is the legacy
        rank-0-only eval pattern.

        For non-ZeRO setups this is a no-op (every rank already has the full
        model).
        """

        # Detect DeepSpeed ZeRO-3. The most reliable signal is that one of the
        # parameters carries a ``ds_id`` attribute (added by the DeepSpeed
        # partition engine). If so we import lazily and use GatheredParameters.
        try:
            has_ds_params = any(
                hasattr(p, "ds_id") for p in model.parameters()
            )
        except Exception:
            has_ds_params = False

        if not has_ds_params:
            return _NullContext()

        try:
            import deepspeed
        except ImportError:  # pragma: no cover - deepspeed should be installed
            return _NullContext()

        return deepspeed.zero.GatheredParameters(
            list(model.parameters()),
            modifier_rank=target_rank,
            enabled=True,
        )

    @staticmethod
    def _broadcast_metrics_dict(
        metrics: Dict[str, float], src: int = 0
    ) -> Dict[str, float]:
        """Broadcast a (small) python dict from ``src`` to all ranks so that
        HF Trainer's log() and callbacks see the same values everywhere.
        No-op outside distributed contexts."""
        if not (dist.is_available() and dist.is_initialized()):
            return metrics
        payload = [metrics if dist.get_rank() == src else None]
        dist.broadcast_object_list(payload, src=src)
        return payload[0] or {}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _append_log(self, metrics: Dict[str, float], trainer) -> None:
        try:
            log_path: Path = self.config.log_path  # type: ignore[assignment]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "step": int(trainer.state.global_step),
                "epoch": float(trainer.state.epoch)
                if trainer.state.epoch is not None
                else None,
                **metrics,
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[skill_eval] failed to write log: {exc}")


class MultiDomainSkillGenerationEvaluator:
    """Run existing skill evaluators sequentially and namespace their metrics.

    Each child keeps the exact single-domain scoring and distributed execution
    path.  This wrapper only prevents metric collisions when the same skills
    are evaluated on multiple corpora (for example GOAT and OVON).
    """

    def __init__(
        self,
        evaluators_by_domain: Dict[str, SkillGenerationEvaluator],
    ) -> None:
        if not evaluators_by_domain:
            raise ValueError("At least one domain evaluator is required.")

        normalized: Dict[str, SkillGenerationEvaluator] = {}
        for raw_domain, evaluator in evaluators_by_domain.items():
            domain = re.sub(
                r"[^a-z0-9_]+",
                "_",
                str(raw_domain).strip().lower(),
            ).strip("_")
            if not domain:
                raise ValueError(
                    f"Invalid empty eval domain after normalization: {raw_domain!r}"
                )
            if domain in normalized:
                raise ValueError(f"Duplicate eval domain: {domain!r}")
            normalized[domain] = evaluator
        self.evaluators_by_domain = normalized

    @staticmethod
    def _domain_metric_key(domain: str, key: str) -> str:
        if key.startswith("eval_"):
            return f"eval_{domain}_{key[len('eval_') :]}"
        if key.startswith("train_"):
            return f"train_{domain}_{key[len('train_') :]}"
        return f"{domain}_{key}"

    def run(self, trainer) -> Dict[str, float]:
        started = time.time()
        merged: Dict[str, float] = {}

        for domain, evaluator in self.evaluators_by_domain.items():
            if _is_main_process():
                print(f"[skill_eval] ===== domain: {domain} =====", flush=True)
            domain_metrics = evaluator.run(trainer=trainer)
            for key, value in domain_metrics.items():
                namespaced = self._domain_metric_key(domain, key)
                if namespaced in merged:
                    raise RuntimeError(
                        f"Duplicate multi-domain eval metric: {namespaced}"
                    )
                merged[namespaced] = value

        # Keep the conventional aggregate runtime key expected by Trainer while
        # retaining each child's eval_<domain>_runtime metric.
        merged["eval_runtime"] = time.time() - started
        return merged


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------------------------------------------------------------------------
# Val loading helpers
# ---------------------------------------------------------------------------


def _load_jsonl_split(
    val_dir: Path,
    skill: str,
    suffix: str,
) -> List[dict]:
    """Shared loader for navagent_<skill>_<suffix>.jsonl. ``suffix`` is one
    of {"val", "train"}. Returns [] when the file does not exist (callers
    decide whether that's fatal)."""
    path = val_dir / f"navagent_{skill}_{suffix}.jsonl"
    if not path.exists():
        if _is_main_process():
            print(
                f"[skill_eval] {suffix} file missing for {skill}: {path} — "
                "skipping this skill.",
                flush=True,
            )
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("data_path", "")
            row.setdefault("dataset_name", f"navagent_{skill}_{suffix}")
            row["skill"] = _normalize_skill_name(row.get("skill") or skill)
            rows.append(row)
    if _is_main_process():
        print(
            f"[skill_eval] loaded {len(rows)} {suffix} samples for {skill}",
            flush=True,
        )
    return rows


def load_val_datasets(
    val_dir: Path,
    skills: Tuple[str, ...] = ("estimating", "planning"),
) -> Dict[str, List[dict]]:
    """Reads navagent_<skill>_val.jsonl files and attaches ``data_path`` /
    ``dataset_name`` / ``skill`` fields the same way LazySupervisedDataset
    would, so that downstream helpers (_build_messages, _normalize_skill_name)
    behave consistently.
    """
    return {skill: _load_jsonl_split(val_dir, skill, "val") for skill in skills}


def load_train_probe_datasets(
    val_dir: Path,
    skills: Tuple[str, ...] = ("estimating", "planning"),
    max_per_skill: int = 1024,
) -> Dict[str, List[dict]]:
    """Build the train-probe pool for each skill.

    Per-skill resolution:
      1. If ``navagent_<skill>_train_probe.jsonl`` exists, load it as-is.
         The splitter's stratified balanced mode produces this file
         already class-balanced (e.g. 96 Normal + 96 Completed), so no
         further subsampling is needed.
      2. Otherwise fall back to the head of ``navagent_<skill>_train.jsonl``
         capped at ``max_per_skill``. Class balance follows the natural
         train distribution.
    """
    out: Dict[str, List[dict]] = {}
    for skill in skills:
        probe_path = val_dir / f"navagent_{skill}_train_probe.jsonl"
        if probe_path.exists():
            probe_rows = _load_jsonl_split(val_dir, skill, "train_probe")
            if _is_main_process():
                print(
                    f"[skill_eval] using dedicated train_probe file for "
                    f"{skill}: {len(probe_rows)} samples (no subsampling).",
                    flush=True,
                )
            out[skill] = probe_rows
            continue
        rows = _load_jsonl_split(val_dir, skill, "train")
        if max_per_skill > 0 and len(rows) > max_per_skill:
            # Deterministic head — _pick_subset will further sample inside.
            # We trim from the head rather than random-sample because the
            # train jsonl is already shuffled at export time, and we want
            # this loader to be cheap and reproducible without extra seeding
            # surface.
            rows = rows[:max_per_skill]
            if _is_main_process():
                print(
                    f"[skill_eval] capped {skill} train probe pool to "
                    f"{max_per_skill} (full file is larger).",
                    flush=True,
                )
        out[skill] = rows
    return out
