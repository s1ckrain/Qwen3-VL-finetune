"""Plot total and per-skill loss curves from skill_loss_history.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot total and per-skill loss curves from skill_loss_history.jsonl."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to skill_loss_history.jsonl",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output image path. Defaults to <input_dir>/loss_curves.png",
    )
    parser.add_argument(
        "--title",
        default="NavAgent Skill Loss Curves",
        help="Plot title.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "loss_curves.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    series = {
        "loss_total": {"x": [], "y": [], "label": "total"},
        "loss_observing": {"x": [], "y": [], "label": "observing"},
        "loss_estimating": {"x": [], "y": [], "label": "estimating"},
        "loss_scheduler": {"x": [], "y": [], "label": "scheduler"},
        "loss_planning": {"x": [], "y": [], "label": "planning"},
    }

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            step = record.get("step")
            if step is None:
                continue
            for key, payload in series.items():
                if key in record:
                    payload["x"].append(step)
                    payload["y"].append(record[key])

    plt.figure(figsize=(10, 6))
    for key, payload in series.items():
        if not payload["x"]:
            continue
        linewidth = 2.2 if key == "loss_total" else 1.8
        alpha = 1.0 if key == "loss_total" else 0.9
        plt.plot(
            payload["x"],
            payload["y"],
            marker="o",
            markersize=3,
            linewidth=linewidth,
            alpha=alpha,
            label=payload["label"],
        )

    plt.title(args.title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
