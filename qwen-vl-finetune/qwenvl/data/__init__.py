import os
import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

NAVAGENT_QWENVL_DATA_DIR = os.environ.get(
    "NAVAGENT_QWENVL_DATA_DIR", "/root/data1/SFT/qwenvl"
)

def _navagent_dataset(filename: str) -> dict:
    return {
        "annotation_path": f"{NAVAGENT_QWENVL_DATA_DIR}/{filename}",
        "data_path": "",
    }


NAVAGENT_OBSERVING = _navagent_dataset("navagent_observing.jsonl")
NAVAGENT_ESTIMATING = _navagent_dataset("navagent_estimating.jsonl")
NAVAGENT_SCHEDULER = _navagent_dataset("navagent_scheduler.jsonl")
NAVAGENT_PLANNING = _navagent_dataset("navagent_planning.jsonl")

# Episode-level train/val split produced by tools/split_navagent_val.py.
# The `_train` variants are what you should point training at; the `_val`
# variants are consumed by the SkillGenerationEvaluator callback.
NAVAGENT_OBSERVING_TRAIN = _navagent_dataset("navagent_observing_train.jsonl")
NAVAGENT_ESTIMATING_TRAIN = _navagent_dataset("navagent_estimating_train.jsonl")
NAVAGENT_SCHEDULER_TRAIN = _navagent_dataset("navagent_scheduler_train.jsonl")
NAVAGENT_PLANNING_TRAIN = _navagent_dataset("navagent_planning_train.jsonl")

NAVAGENT_OBSERVING_VAL = _navagent_dataset("navagent_observing_val.jsonl")
NAVAGENT_ESTIMATING_VAL = _navagent_dataset("navagent_estimating_val.jsonl")
NAVAGENT_SCHEDULER_VAL = _navagent_dataset("navagent_scheduler_val.jsonl")
NAVAGENT_PLANNING_VAL = _navagent_dataset("navagent_planning_val.jsonl")

# --- Joint GOAT + OVON SFT datasets -----------------------------------------
# Keep both corpora separate on disk. Their rows share the same Qwen-VL schema
# and use absolute image paths, so LazySupervisedDataset can concatenate them.
NAVAGENT_GOAT_QWENVL_DATA_DIR = os.environ.get(
    "NAVAGENT_GOAT_QWENVL_DATA_DIR", "/root/data1/SFT-goat/qwenvl"
)
NAVAGENT_OVON_QWENVL_DATA_DIR = os.environ.get(
    "NAVAGENT_OVON_QWENVL_DATA_DIR", "/root/data1/SFT-ovon/qwenvl"
)


def _navagent_dataset_from(data_dir: str, filename: str) -> dict:
    return {
        "annotation_path": f"{data_dir}/{filename}",
        "data_path": "",
    }


NAVAGENT_GOAT_OBSERVING_TRAIN = _navagent_dataset_from(
    NAVAGENT_GOAT_QWENVL_DATA_DIR, "navagent_observing_train.jsonl"
)
NAVAGENT_GOAT_ESTIMATING_TRAIN = _navagent_dataset_from(
    NAVAGENT_GOAT_QWENVL_DATA_DIR, "navagent_estimating_train.jsonl"
)
NAVAGENT_GOAT_PLANNING_TRAIN = _navagent_dataset_from(
    NAVAGENT_GOAT_QWENVL_DATA_DIR, "navagent_planning_train.jsonl"
)

NAVAGENT_OVON_OBSERVING_TRAIN = _navagent_dataset_from(
    NAVAGENT_OVON_QWENVL_DATA_DIR, "navagent_observing_train.jsonl"
)
NAVAGENT_OVON_ESTIMATING_TRAIN = _navagent_dataset_from(
    NAVAGENT_OVON_QWENVL_DATA_DIR, "navagent_estimating_train.jsonl"
)
NAVAGENT_OVON_PLANNING_TRAIN = _navagent_dataset_from(
    NAVAGENT_OVON_QWENVL_DATA_DIR, "navagent_planning_train.jsonl"
)

# --- DAgger (Stage-2) datasets ---------------------------------------------
# Separate directory so Stage-2 student-rollout data never mixes with the SFT
# qwenvl dir. Kept as distinct dataset keys (`*_dagger`) so launchers can train
# on DAgger only or mix it explicitly for ablations.
NAVAGENT_DAGGER_DATA_DIR = os.environ.get(
    "NAVAGENT_DAGGER_DATA_DIR", "/root/data1/SFT-v4/dagger_qwenvl"
)


def _navagent_dagger_dataset(filename: str) -> dict:
    return {
        "annotation_path": f"{NAVAGENT_DAGGER_DATA_DIR}/{filename}",
        "data_path": "",
    }


NAVAGENT_OBSERVING_DAGGER = _navagent_dagger_dataset("navagent_observing.jsonl")
NAVAGENT_ESTIMATING_DAGGER = _navagent_dagger_dataset("navagent_estimating.jsonl")
NAVAGENT_PLANNING_DAGGER = _navagent_dagger_dataset("navagent_planning.jsonl")

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "navagent_observing": NAVAGENT_OBSERVING,
    "navagent_estimating": NAVAGENT_ESTIMATING,
    "navagent_scheduler": NAVAGENT_SCHEDULER,
    "navagent_planning": NAVAGENT_PLANNING,
    "navagent_observing_train": NAVAGENT_OBSERVING_TRAIN,
    "navagent_estimating_train": NAVAGENT_ESTIMATING_TRAIN,
    "navagent_scheduler_train": NAVAGENT_SCHEDULER_TRAIN,
    "navagent_planning_train": NAVAGENT_PLANNING_TRAIN,
    "navagent_observing_val": NAVAGENT_OBSERVING_VAL,
    "navagent_estimating_val": NAVAGENT_ESTIMATING_VAL,
    "navagent_scheduler_val": NAVAGENT_SCHEDULER_VAL,
    "navagent_planning_val": NAVAGENT_PLANNING_VAL,
    "navagent_goat_observing_train": NAVAGENT_GOAT_OBSERVING_TRAIN,
    "navagent_goat_estimating_train": NAVAGENT_GOAT_ESTIMATING_TRAIN,
    "navagent_goat_planning_train": NAVAGENT_GOAT_PLANNING_TRAIN,
    "navagent_ovon_observing_train": NAVAGENT_OVON_OBSERVING_TRAIN,
    "navagent_ovon_estimating_train": NAVAGENT_OVON_ESTIMATING_TRAIN,
    "navagent_ovon_planning_train": NAVAGENT_OVON_PLANNING_TRAIN,
    "navagent_observing_dagger": NAVAGENT_OBSERVING_DAGGER,
    "navagent_estimating_dagger": NAVAGENT_ESTIMATING_DAGGER,
    "navagent_planning_dagger": NAVAGENT_PLANNING_DAGGER,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
