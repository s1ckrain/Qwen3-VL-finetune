import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    dataset_resample_weights: str = field(
        default="",
        metadata={
            "help": (
                "Comma-separated dataset or skill weights, e.g. "
                "'observing=1,estimating=1,scheduler=1,planning=1' or "
                "'navagent_observing=1,navagent_planning=2'."
            )
        },
    )
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2

    # --- Skill generation eval (periodic success-rate check) ----------------
    eval_val_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory containing navagent_<skill>_val.jsonl files produced "
                "by tools/split_navagent_val.py. When set, a generation-based "
                "eval callback is attached to the trainer."
            )
        },
    )
    eval_skills: str = field(
        default="estimating,planning",
        metadata={
            "help": "Comma-separated list of skills to generate+score on val."
        },
    )
    eval_num_samples_per_skill: int = field(
        default=300,
        metadata={
            "help": "Max val samples evaluated per skill per eval cycle."
        },
    )
    eval_max_new_tokens: int = field(
        default=512,
        metadata={"help": "max_new_tokens passed to model.generate() during eval."},
    )
    eval_train_probe_size: int = field(
        default=0,
        metadata={
            "help": (
                "If > 0, ALSO probe the train split each eval cycle by "
                "running generate on this many fixed samples drawn from "
                "navagent_<skill>_train.jsonl. The gap between train_acc "
                "and val_acc tells us whether the model is fitting the "
                "data at all (vs. just generalizing). Set to 0 to disable."
            )
        },
    )
    eval_planning_pixel_tolerance: float = field(
        default=64.0,
        metadata={
            "help": (
                "L2 (Euclidean) pixel tolerance for planning's target_pixel "
                "during periodic eval. A predicted pixel counts as 'pixel "
                "correct' iff its L2 distance to the oracle pixel is strictly "
                "less than this many pixels. Default 64 ≈ 10% of the 640px "
                "image width. Joint accuracy (the headline planning metric) "
                "additionally requires direction to match."
            )
        },
    )

    # --- Per-skill loss weighting (scales compute_loss by skill) ------------
    skill_loss_weights: str = field(
        default="",
        metadata={
            "help": (
                "Comma-separated per-skill loss weights, e.g. "
                "'observing=1,estimating=2,scheduler=2,planning=2'. Weights are "
                "normalized so that mean(weights)=1 across the listed skills, "
                "which keeps the effective learning rate constant while changing "
                "the relative gradient contribution of each skill. Leave empty "
                "to disable (all skills weighted equally)."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
