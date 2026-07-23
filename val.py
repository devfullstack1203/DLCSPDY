#!/usr/bin/env python3
"""Evaluate DLCSPDY checkpoints on val or test splits.

Usage:
    # Validate on val split (default)
    python val.py --weights runs/detect/train/weights/best.pt --data configs/datasets/ppe2025.yaml

    # Evaluate on test split
    python val.py --weights runs/detect/train/weights/best.pt --data configs/datasets/ppe2025.yaml --split test

    # Dataset shortcut
    python val.py --weights runs/detect/PPEyolov13n/weights/best.pt --dataset ppe2025 --split test
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from ultralytics import YOLO

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "ppe2025": ROOT / "configs" / "datasets" / "ppe2025.yaml",
    "chvg": ROOT / "configs" / "datasets" / "chvg.yaml",
    "cppe": ROOT / "configs" / "datasets" / "cppe.yaml",
}


def resolve_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def resolve_data(args: argparse.Namespace) -> str:
    if args.data:
        return resolve_path(args.data)
    if args.dataset:
        return str(DATASETS[args.dataset])
    raise SystemExit("Provide --data or --dataset (ppe2025 | chvg | cppe).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate / test DLCSPDY detectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (.pt)",
    )
    parser.add_argument("--data", default=None, help="Dataset YAML (overrides --dataset)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument(
        "--split",
        choices=("val", "test", "train"),
        default="val",
        help="Dataset split to evaluate",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="GPU id, e.g. 0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no-plots", action="store_false", dest="plots")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = resolve_path(args.weights)
    data_path = resolve_data(args)

    if not Path(weights).is_file():
        raise SystemExit(f"Weights not found: {weights}")

    print("=" * 55)
    print("DLCSPDY Evaluation")
    print("=" * 55)
    print(f"  Weights : {weights}")
    print(f"  Data    : {data_path}")
    print(f"  Split   : {args.split}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print("=" * 55)

    model = YOLO(weights)
    kwargs = {
        "data": data_path,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "plots": args.plots,
        "save_json": args.save_json,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    model.val(**kwargs)


if __name__ == "__main__":
    main()
