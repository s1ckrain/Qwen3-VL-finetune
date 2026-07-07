"""Split the exported NavAgent QwenVL jsonl files into train/val (+ optional
train probe).

Two split modes:

  * **episode-level** (default for all skills): hold out whole episodes
    keyed by ``(scene_id, episode_id, sub_episode_index)`` so the val set
    never shares images or trajectory context with train. Greedy episode
    picker keeps adding shuffled episodes until each primary skill (the
    ones in ``PRIMARY_SKILLS``) reaches the per-skill target.

  * **stratified step-level** (opt-in via ``--stratify-balanced-skill``,
    supported for ``estimating`` and ``planning``): pick a fixed *exactly*
    balanced val + train_probe set directly from the raw step pool.
      - estimating: ``--balance-per-class`` Normal + Completed each, for
        val and train_probe (default 96 ⇒ 192 / 192).
      - planning : ``--balance-per-class`` front+right+left+back each, for
        val and train_probe (default 48 ⇒ 192 / 192).
    The rest goes to train. This trades a little frame-level cleanliness
    for guaranteed per-class denominators so per-class recall numbers are
    statistically tight (n=48 ⇒ ±14pp 95% CI on Bernoulli).

  ``--balance-per-class`` may be passed multiple times in
  ``skill=N`` form to give each stratified skill its own per-class size
  (e.g. ``--balance-per-class estimating=96 --balance-per-class
  planning=48``). A bare integer (``--balance-per-class 96``) sets the
  same value for every stratified skill, matching the old behaviour.

Outputs (under --output-dir, default = input dir):
  navagent_<skill>_train.jsonl       : training samples
  navagent_<skill>_val.jsonl         : held-out validation samples
  navagent_<skill>_train_probe.jsonl : (only when stratified) fixed
                                       balanced subset drawn from train
                                       for use as a "is the model fitting
                                       at all?" probe during eval cycles
  navagent_val_manifest.json         : split bookkeeping

The script is idempotent: if all output files already exist and the manifest
matches the requested config, it exits without rewriting anything (use
--force to override).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SKILLS: Tuple[str, ...] = ("observing", "estimating", "scheduler", "planning")
# Skills whose validation accuracy we actually care about. The held-out
# episodes are picked so that *these* skills reach the per-skill target; the
# other skills just carry along whatever fraction of their samples happen to
# fall in the same episodes.
PRIMARY_SKILLS: Tuple[str, ...] = ("estimating", "planning")

# Skills for which step-level class-balanced split is supported. Each requires
# a custom label extractor in ``_BALANCE_LABEL_FNS`` below.
STRATIFY_SUPPORTED: Tuple[str, ...] = ("estimating", "planning")

EpisodeKey = Tuple[str, str, int]


def _parse_gpt_json(sample: dict) -> dict | None:
    """Shared helper: load the gpt turn (last conversation entry) as JSON.

    Tolerates a leading markdown fence in case any teacher output sneaked
    one through (the export step strips them, but be defensive).
    """
    convs = sample.get("conversations") or []
    if not convs:
        return None
    gpt = convs[-1].get("value", "") if isinstance(convs[-1], dict) else ""
    if not gpt:
        return None
    text = gpt.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_estimating_label(sample: dict) -> str | None:
    """Canonical status string out of an exported estimating row."""
    obj = _parse_gpt_json(sample)
    if obj is None:
        return None
    status = obj.get("status")
    if not isinstance(status, str):
        return None
    canon = status.strip().lower()
    if canon == "normal":
        return "Normal"
    if canon == "completed":
        return "Completed"
    if canon == "lost":
        return "Lost"
    return None


def _extract_planning_label(sample: dict) -> str | None:
    """Canonical direction out of an exported planning row.

    The four canonical values are stored lowercase ``front/right/left/back``
    which is what NavAgent/skills/planning.py emits and what the eval code
    compares against (case-insensitive). We return that casing.
    """
    obj = _parse_gpt_json(sample)
    if obj is None:
        return None
    raw = obj.get("selected_direction")
    if not isinstance(raw, str):
        return None
    canon = raw.strip().lower()
    if canon in ("front", "right", "left", "back"):
        return canon
    return None


_BALANCE_LABEL_FNS = {
    "estimating": _extract_estimating_label,
    "planning": _extract_planning_label,
}
# Per-skill default class set for balanced sampling. Lost is excluded for
# estimating because the labeled corpus historically has zero Lost samples;
# including it would force the picker to fall short of per-class target.
_BALANCE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "estimating": ("Normal", "Completed"),
    "planning": ("front", "right", "left", "back"),
}
# Per-skill default per-class size for stratified split when the user does
# not pin a value explicitly. Picked so val/train_probe each land at 192
# samples total per skill (matches what the eval pipeline budgets for).
_BALANCE_PER_CLASS_DEFAULTS: Dict[str, int] = {
    "estimating": 96,   # 96 * 2  = 192 per split
    "planning":   48,   # 48 * 4  = 192 per split
}

# Per-skill default sub-stratification values used when the caller opts into
# sub-stratification via ``--sub-stratify-skill skill=field``.  The split
# picker then evenly distributes ``balance_per_class`` across these values
# within each primary class.  Order is fixed so splits are reproducible.
_SUB_STRATIFY_VALUES: Dict[str, Tuple[str, ...]] = {
    "estimating": ("object", "description", "image"),
    "planning":   ("object", "description", "image"),
}


def _read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _episode_key(sample: dict) -> EpisodeKey | None:
    meta = sample.get("meta") or {}
    scene = meta.get("scene_id")
    episode = meta.get("episode_id")
    sub = meta.get("sub_episode_index", 0)
    if scene is None or episode is None:
        return None
    try:
        sub_int = int(sub)
    except (TypeError, ValueError):
        sub_int = 0
    return (str(scene), str(episode), sub_int)


def _stable_shuffle(items: List[EpisodeKey], seed: int) -> List[EpisodeKey]:
    """Deterministic shuffle that is stable across runs regardless of Python
    hash seed. We hash each tuple to a 64-bit int, then sort by (hash, seed).
    """

    def _score(key: EpisodeKey) -> int:
        payload = f"{seed}|{key[0]}|{key[1]}|{key[2]}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big")

    return sorted(items, key=_score)


def _load_all(input_dir: Path) -> Dict[str, List[dict]]:
    """Load every ``navagent_<skill>.jsonl`` that exists in *input_dir*.

    Skills whose source file is absent are skipped with a notice.  This
    accommodates the new pipeline where Scheduler is no longer labelled
    by default (see NavAgent/sft-label.sh).  Callers that need a
    specific skill should check ``skill in datasets`` themselves.
    """
    datasets: Dict[str, List[dict]] = {}
    for skill in SKILLS:
        path = input_dir / f"navagent_{skill}.jsonl"
        if not path.exists():
            print(
                f"[split_navagent_val] skipping missing source jsonl: {path}",
                file=sys.stderr,
            )
            continue
        datasets[skill] = _read_jsonl(path)
    if not datasets:
        raise FileNotFoundError(
            f"no navagent_<skill>.jsonl files found under {input_dir}. "
            "run prepare_navagent_data.sh first."
        )
    return datasets


def _pick_val_episodes(
    datasets: Dict[str, List[dict]],
    target_per_skill: int,
    seed: int,
    primary_skills: Tuple[str, ...] = PRIMARY_SKILLS,
) -> Tuple[List[EpisodeKey], Dict[str, int]]:
    """Greedy episode picker that keeps adding shuffled episodes until each
    skill in ``primary_skills`` has at least ``target_per_skill`` val samples.

    The same episode set is applied across ALL skills present in ``datasets``
    so train/val don't share any observation (scene_id, episode_id,
    sub_episode_index). Skills not present in ``datasets`` are ignored —
    callers using a stratified path for some skills should pass only the
    remaining skills here.
    """

    # Index: episode_key -> {skill: sample_count}, restricted to skills
    # actually present in ``datasets``.
    present_skills = tuple(datasets.keys())
    per_skill_counts: Dict[EpisodeKey, Dict[str, int]] = {}
    for skill, rows in datasets.items():
        for row in rows:
            key = _episode_key(row)
            if key is None:
                continue
            per_skill_counts.setdefault(
                key, {s: 0 for s in present_skills}
            )[skill] += 1

    if not per_skill_counts:
        return [], {s: 0 for s in present_skills}

    episodes = list(per_skill_counts.keys())
    shuffled = _stable_shuffle(episodes, seed=seed)

    running = {skill: 0 for skill in present_skills}
    picked: List[EpisodeKey] = []

    # Only require coverage for primary skills that are actually in datasets.
    effective_primary = tuple(s for s in primary_skills if s in present_skills)

    def primary_filled() -> bool:
        if not effective_primary:
            return True
        return all(running[s] >= target_per_skill for s in effective_primary)

    for key in shuffled:
        if primary_filled():
            break
        counts = per_skill_counts[key]
        picked.append(key)
        for skill in present_skills:
            running[skill] += counts[skill]

    if not primary_filled():
        print(
            "[split_navagent_val] WARNING: exhausted episodes before meeting "
            f"target {target_per_skill} per primary skill. Final counts: {running}",
            file=sys.stderr,
        )

    return picked, running


def _partition_and_write(
    datasets: Dict[str, List[dict]],
    held_out: set,
    output_dir: Path,
) -> Dict[str, Dict[str, int]]:
    sizes: Dict[str, Dict[str, int]] = {}
    for skill, rows in datasets.items():
        train_rows: List[dict] = []
        val_rows: List[dict] = []
        for row in rows:
            key = _episode_key(row)
            if key is not None and key in held_out:
                val_rows.append(row)
            else:
                train_rows.append(row)
        _write_jsonl(output_dir / f"navagent_{skill}_train.jsonl", train_rows)
        _write_jsonl(output_dir / f"navagent_{skill}_val.jsonl", val_rows)
        sizes[skill] = {"train": len(train_rows), "val": len(val_rows)}
    return sizes


def _manifest_matches(
    manifest_path: Path,
    episodes: List[EpisodeKey],
    target_per_skill: int,
    seed: int,
    stratify_skills: Tuple[str, ...] = (),
    balance_per_class_map: Optional[Dict[str, int]] = None,
    sub_stratify_map: Optional[Dict[str, str]] = None,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("target_per_skill") != target_per_skill:
        return False
    if data.get("seed") != seed:
        return False
    if tuple(data.get("stratify_skills", [])) != tuple(stratify_skills):
        return False
    # New (multi-skill) format: balance_per_class_map. Legacy format had a
    # single int under ``balance_per_class``. Accept both for back-compat
    # so re-running an old manifest doesn't force a needless rebuild when
    # there's only one stratified skill.
    expected_map = balance_per_class_map or {}
    if "balance_per_class_map" in data:
        if dict(data.get("balance_per_class_map") or {}) != expected_map:
            return False
    else:
        legacy = data.get("balance_per_class", 0)
        if len(expected_map) > 1:
            return False
        if expected_map and next(iter(expected_map.values())) != legacy:
            return False
        if not expected_map and legacy != 0:
            return False
    # sub_stratify_map: missing in legacy manifests is treated as {} —
    # any non-empty current value forces a rebuild.
    expected_sub = sub_stratify_map or {}
    existing_sub = dict(data.get("sub_stratify_map") or {})
    if existing_sub != expected_sub:
        return False
    existing = {tuple(ep) for ep in data.get("episodes", [])}
    return existing == set(episodes)


def _stratified_balanced_split(
    skill: str,
    rows: List[dict],
    output_dir: Path,
    *,
    balance_per_class: int,
    seed: int,
    sub_stratify_field: Optional[str] = None,
) -> Dict[str, int]:
    """Build a class-balanced (val, train, train_probe) split for one skill.

    Layout:
      * val gets ``balance_per_class`` samples from each class in
        ``_BALANCE_CLASSES[skill]``. val rows are *not* in train.
      * train gets every remaining row (including rows skipped because their
        label was outside the target class set, e.g. Lost for estimating).
      * train_probe is a class-balanced *subset of train* (also
        ``balance_per_class`` per class). Probe rows therefore ARE seen
        during training; their accuracy is the "is the model fitting at
        all?" diagnostic. The val ↔ probe gap reflects overfitting.

    ``sub_stratify_field`` (optional) names a categorical field on the
    record's ``meta`` dict (e.g. ``"goal_type"``).  When provided, each
    primary-class bucket is further partitioned across the per-skill
    ``_SUB_STRATIFY_VALUES`` and ``balance_per_class`` is distributed
    evenly across the sub-buckets.  E.g. estimating with
    ``balance_per_class=96`` and ``sub_stratify_field="goal_type"`` yields
    32 samples per (Normal|Completed) × (object|description|image)
    bucket — 192 total per split, exactly 1:1:1 across goal types.

    Picks are stable across runs (per-row hash salted by seed + class +
    sub_value) so re-prep is reproducible and only newly-added rows can shift.
    """
    label_fn = _BALANCE_LABEL_FNS.get(skill)
    if label_fn is None:
        raise ValueError(f"No label extractor registered for skill={skill}")

    classes = _BALANCE_CLASSES[skill]

    # Sub-stratification setup.
    if sub_stratify_field:
        sub_values = _SUB_STRATIFY_VALUES.get(skill)
        if not sub_values:
            raise ValueError(
                f"sub_stratify_field={sub_stratify_field!r} requested for "
                f"skill={skill} but no _SUB_STRATIFY_VALUES entry exists."
            )
        per_sub_target = balance_per_class // len(sub_values)
    else:
        sub_values = ("",)
        per_sub_target = balance_per_class

    def _row_sub_value(row: dict) -> str:
        if not sub_stratify_field:
            return ""
        return str(row.get("meta", {}).get(sub_stratify_field, "") or "")

    # Bucket rows by (label, sub_value).
    buckets: Dict[Tuple[str, str], List[Tuple[int, dict]]] = {
        (c, sv): [] for c in classes for sv in sub_values
    }
    skipped = 0
    for idx, row in enumerate(rows):
        label = label_fn(row)
        if label is None or label not in classes:
            skipped += 1
            continue
        sub_value = _row_sub_value(row)
        if sub_stratify_field and sub_value not in sub_values:
            skipped += 1
            continue
        buckets[(label, sub_value)].append((idx, row))

    # Stable per-bucket shuffle (hash on episode key + sample idx, salted by
    # seed + class + sub_value so val/probe assignments don't correlate
    # across buckets).
    def _shuffle_bucket(
        label: str, sub_value: str, items: List[Tuple[int, dict]]
    ):
        def _score(pair: Tuple[int, dict]) -> int:
            idx, row = pair
            ep = _episode_key(row)
            payload = f"{seed}|{label}|{sub_value}|{ep}|{idx}".encode("utf-8")
            return int.from_bytes(
                hashlib.blake2b(payload, digest_size=8).digest(), "big"
            )

        return sorted(items, key=_score)

    val_rows: List[dict] = []
    train_rows: List[dict] = []
    probe_rows: List[dict] = []
    val_idx_set: set = set()

    for label in classes:
        for sub_value in sub_values:
            ordered = _shuffle_bucket(label, sub_value, buckets[(label, sub_value)])
            if len(ordered) < 2 * per_sub_target:
                bucket_desc = f"{skill}/{label}" + (
                    f"/{sub_value}" if sub_stratify_field else ""
                )
                print(
                    f"[split_navagent_val] WARNING: {bucket_desc} has only "
                    f"{len(ordered)} samples — need ≥ {2 * per_sub_target} "
                    f"for val + train_probe at per_sub_target="
                    f"{per_sub_target} (val will use the head; probe pulls "
                    f"from the train tail and may underfill).",
                    file=sys.stderr,
                )
            # Head -> val (excluded from train).
            val_chunk = ordered[:per_sub_target]
            # Everything after the val head goes into train.
            train_class_chunk = ordered[per_sub_target:]
            # Probe is a balanced view OF train: pick the first per-sub-target
            # train rows. Those rows are also in train_rows — by design, so
            # probe ⊂ train.
            probe_chunk = train_class_chunk[:per_sub_target]

            val_rows.extend(row for _, row in val_chunk)
            train_rows.extend(row for _, row in train_class_chunk)
            probe_rows.extend(row for _, row in probe_chunk)
            val_idx_set.update(idx for idx, _ in val_chunk)

    # Rows skipped above (label not parseable, or label outside the target
    # class set — e.g. Lost for estimating; or sub_value outside the target
    # sub-set) go into train so we don't lose data. They are NOT eligible
    # for the probe (probe is balanced over the target buckets only).
    for idx, row in enumerate(rows):
        if idx in val_idx_set:
            continue
        label = label_fn(row)
        if label is not None and label in classes:
            sub_value = _row_sub_value(row)
            if not sub_stratify_field or sub_value in sub_values:
                # Already added to train above as part of train_class_chunk.
                continue
        train_rows.append(row)

    # Shuffle train so the file isn't class-clumped.
    train_indexed = list(enumerate(train_rows))
    train_indexed.sort(
        key=lambda p: int.from_bytes(
            hashlib.blake2b(
                f"{seed}|train|{p[0]}".encode("utf-8"), digest_size=8
            ).digest(),
            "big",
        )
    )
    train_rows = [r for _, r in train_indexed]

    _write_jsonl(output_dir / f"navagent_{skill}_train.jsonl", train_rows)
    _write_jsonl(output_dir / f"navagent_{skill}_val.jsonl", val_rows)
    _write_jsonl(output_dir / f"navagent_{skill}_train_probe.jsonl", probe_rows)

    print(
        f"[split_navagent_val] {skill} stratified: "
        f"train={len(train_rows)} val={len(val_rows)} "
        f"train_probe={len(probe_rows)} skipped={skipped}"
    )
    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "train_probe": len(probe_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/root/data1/SFT/qwenvl"),
        help="Directory containing the source navagent_<skill>.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write split files into. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--target-per-skill",
        type=int,
        default=300,
        help="Target number of val samples for each primary skill "
        f"({', '.join(PRIMARY_SKILLS)}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite outputs even if a matching manifest already exists.",
    )
    parser.add_argument(
        "--stratify-balanced-skill",
        action="append",
        default=[],
        choices=list(STRATIFY_SUPPORTED),
        help=(
            "Skill name to use class-balanced step-level split for instead of "
            "the default episode-level split. Generates an additional "
            "navagent_<skill>_train_probe.jsonl. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--balance-per-class",
        action="append",
        default=[],
        help=(
            "Per-class sample count for stratified skills. Two forms:\n"
            "  * bare integer (e.g. 96): apply this value to every "
            "    stratified skill.\n"
            "  * skill=N (e.g. planning=48): pin per-class size for one "
            "    specific skill. May be passed multiple times.\n"
            "If not set for a given stratified skill, falls back to the "
            "per-skill default in _BALANCE_PER_CLASS_DEFAULTS (estimating=96, "
            "planning=48)."
        ),
    )
    parser.add_argument(
        "--sub-stratify-skill",
        action="append",
        default=[],
        help=(
            "Further partition a stratified skill's per-class budget across "
            "a categorical meta field (e.g. estimating=goal_type). Within "
            "each primary class the per-class total is distributed evenly "
            "across the values listed in _SUB_STRATIFY_VALUES for that skill "
            "(estimating: object/description/image; planning: same). "
            "Form: skill=field. May be passed multiple times."
        ),
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir
    stratify_skills: Tuple[str, ...] = tuple(sorted(set(args.stratify_balanced_skill)))

    # Resolve --balance-per-class into a per-skill dict. Parse rules:
    #   * `skill=N` form overrides only that skill.
    #   * Any bare integer is treated as a global default for *all*
    #     stratified skills that don't have an explicit override.
    #   * If still unset after both passes, fall back to _BALANCE_PER_CLASS_DEFAULTS.
    explicit_map: Dict[str, int] = {}
    global_value: Optional[int] = None
    for raw in args.balance_per_class:
        if not isinstance(raw, str):
            continue
        if "=" in raw:
            skill_name, _, val_s = raw.partition("=")
            skill_name = skill_name.strip()
            if skill_name not in STRATIFY_SUPPORTED:
                parser.error(
                    f"--balance-per-class refers to unsupported skill: {skill_name!r}"
                )
            try:
                explicit_map[skill_name] = int(val_s.strip())
            except ValueError:
                parser.error(f"--balance-per-class value not an int: {raw!r}")
        else:
            try:
                global_value = int(raw.strip())
            except ValueError:
                parser.error(f"--balance-per-class value not an int: {raw!r}")

    balance_per_class_map: Dict[str, int] = {}
    for skill in stratify_skills:
        if skill in explicit_map:
            balance_per_class_map[skill] = explicit_map[skill]
        elif global_value is not None:
            balance_per_class_map[skill] = global_value
        else:
            balance_per_class_map[skill] = _BALANCE_PER_CLASS_DEFAULTS.get(skill, 0)

    # Parse --sub-stratify-skill into a per-skill map: skill → meta field name.
    sub_stratify_map: Dict[str, str] = {}
    for raw in args.sub_stratify_skill:
        if not isinstance(raw, str) or "=" not in raw:
            parser.error(
                f"--sub-stratify-skill expects skill=field form: {raw!r}"
            )
        skill_name, _, field_name = raw.partition("=")
        skill_name = skill_name.strip()
        field_name = field_name.strip()
        if skill_name not in STRATIFY_SUPPORTED:
            parser.error(
                f"--sub-stratify-skill refers to unsupported skill: {skill_name!r}"
            )
        if skill_name not in _SUB_STRATIFY_VALUES:
            parser.error(
                f"--sub-stratify-skill: no _SUB_STRATIFY_VALUES entry for "
                f"skill={skill_name!r}"
            )
        if not field_name:
            parser.error(
                f"--sub-stratify-skill: empty field name in {raw!r}"
            )
        sub_stratify_map[skill_name] = field_name

    print(f"[split_navagent_val] input_dir={input_dir} output_dir={output_dir}")
    if stratify_skills:
        per_class_summary = ", ".join(
            f"{s}={balance_per_class_map[s]}" for s in stratify_skills
        )
        print(
            f"[split_navagent_val] stratified balanced split for: "
            f"{stratify_skills} (per-class: {per_class_summary})"
        )
    datasets = _load_all(input_dir)
    for skill, rows in datasets.items():
        print(f"[split_navagent_val] loaded {skill}: {len(rows)} samples")

    random.seed(args.seed)  # kept for any downstream consumers of random
    # Episode-level pick is still based on ALL skills' data (so the held-out
    # set is the same across skills) — but we exclude any stratified skill
    # from the "must reach target" check, since it's getting its own split.
    episode_target_datasets = {
        skill: rows for skill, rows in datasets.items() if skill not in stratify_skills
    }
    picked, running = _pick_val_episodes(
        episode_target_datasets or datasets,
        target_per_skill=args.target_per_skill,
        seed=args.seed,
    )
    print(
        f"[split_navagent_val] held out {len(picked)} episodes (episode-level "
        f"skills). val sample counts: {running}"
    )

    manifest_path = output_dir / "navagent_val_manifest.json"
    if not args.force and _manifest_matches(
        manifest_path,
        picked,
        args.target_per_skill,
        args.seed,
        stratify_skills=stratify_skills,
        balance_per_class_map=balance_per_class_map,
        sub_stratify_map=sub_stratify_map,
    ):
        # Still ensure the per-skill files exist; if any are missing we force rewrite.
        # Only check skills that were actually loaded (e.g. Scheduler may be absent).
        required_files: List[Path] = []
        for skill in datasets.keys():
            for split in ("train", "val"):
                required_files.append(
                    output_dir / f"navagent_{skill}_{split}.jsonl"
                )
            if skill in stratify_skills:
                required_files.append(
                    output_dir / f"navagent_{skill}_train_probe.jsonl"
                )
        missing = [p for p in required_files if not p.exists()]
        if not missing:
            print(
                "[split_navagent_val] manifest up to date and all split files exist, "
                "skipping (use --force to rebuild)."
            )
            return

    held_out = set(picked)
    # Episode-level skills first.
    episode_skill_datasets = {
        skill: rows for skill, rows in datasets.items() if skill not in stratify_skills
    }
    sizes = _partition_and_write(episode_skill_datasets, held_out, output_dir)

    # Stratified skills next (independent of the episode-level pick).
    for skill in stratify_skills:
        sz = _stratified_balanced_split(
            skill,
            datasets[skill],
            output_dir,
            balance_per_class=balance_per_class_map[skill],
            seed=args.seed,
            sub_stratify_field=sub_stratify_map.get(skill),
        )
        sizes[skill] = sz

    manifest = {
        "target_per_skill": args.target_per_skill,
        "seed": args.seed,
        "primary_skills": list(PRIMARY_SKILLS),
        "stratify_skills": list(stratify_skills),
        "balance_per_class_map": balance_per_class_map,
        "sub_stratify_map": sub_stratify_map,
        "episodes": [list(ep) for ep in picked],
        "per_skill_counts": {
            skill: sizes[skill] for skill in SKILLS if skill in sizes
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[split_navagent_val] wrote manifest: {manifest_path}")
    for skill, sz in sizes.items():
        extra = f" train_probe={sz['train_probe']}" if "train_probe" in sz else ""
        print(
            f"[split_navagent_val] {skill}: train={sz['train']} val={sz['val']}{extra}"
        )


if __name__ == "__main__":
    main()
