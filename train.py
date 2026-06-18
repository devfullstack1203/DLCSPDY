#!/usr/bin/env python3
"""Train DLCSPDY — mirrors Colab notebooks in Colab_Notebooks/yolov13.

Standard training (YOLOv13 / DY / CSP / CSPDY):
    model = YOLO(variant_yaml)
    model.train(data=..., epochs=200, imgsz=640, batch=32,
                device=0, optimizer='auto', plots=True, save_period=50)

DLCSPDY (knowledge distillation):
    model_t = YOLO(teacher.pt)
    model_t.model.model[-1].set_Distillation = True
    model_s = YOLO(CSPDYyolov13n.yaml)
    model_s.train(..., project='runs/train', workers=0, model_t=model_t.model)

Usage:
    # DLCSPDY on PPE2025 (default)
    python train.py --dataset ppe2025 --teacher runs/detect/PPEyolov13n/weights/best.pt

    # Baselines
    python train.py --variant yolov13 --dataset ppe2025 --name PPEyolov13n
    python train.py --variant dy      --dataset chvg   --name CHVGDYyolo
    python train.py --variant csp     --dataset cppe   --name ENVCSPyolo
    python train.py --variant cspdy   --dataset ppe2025 --name PPECSPDYyolo

    # DLCSPDY per dataset
    python train.py --variant dlcspdy --dataset chvg --teacher path/to/teacher.pt --name CHVGDLCSPDYyolo
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import yaml

from ultralytics import YOLO

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "configs" / "train_default.yaml"
DATASETS = {
    "ppe2025": ROOT / "configs" / "datasets" / "ppe2025.yaml",
    "chvg": ROOT / "configs" / "datasets" / "chvg.yaml",
    "cppe": ROOT / "configs" / "datasets" / "cppe.yaml",
}
VARIANTS = ("yolov13", "dy", "csp", "cspdy", "dlcspdy")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def resolve_data(args: argparse.Namespace, cfg: dict) -> str:
    if args.data:
        return resolve_path(args.data)
    if args.dataset:
        return str(DATASETS[args.dataset])
    return resolve_path(cfg["data"])


def resolve_variant_model(variant: str, cfg: dict, explicit_model: str | None) -> str:
    if explicit_model:
        return resolve_path(explicit_model)
    variants = cfg.get("variants", {})
    if variant in variants:
        return resolve_path(variants[variant])
    return resolve_path(cfg["model"])


def resolve_teacher(args: argparse.Namespace, cfg: dict) -> str | None:
    if args.teacher:
        return resolve_path(args.teacher)
    if not args.dataset:
        return None
    teachers = cfg.get("distill", {}).get("teachers", {})
    teacher = teachers.get(args.dataset)
    return resolve_path(teacher) if teacher else None


def build_train_kwargs(cfg: dict, args: argparse.Namespace, variant: str) -> dict:
    """Build kwargs passed to model.train(), matching Colab notebook style."""
    skip = {"model", "data", "variant", "variants", "distill"}
    train_kwargs = {k: v for k, v in cfg.items() if k not in skip}

    train_kwargs.update(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        seed=args.seed,
        optimizer=cfg.get("optimizer", "auto"),
        plots=cfg.get("plots", True),
        save_period=cfg.get("save_period", 50),
        cache=cfg.get("cache", False),
        single_cls=cfg.get("single_cls", False),
    )

    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    if args.resume:
        train_kwargs["resume"] = True
    if args.device is not None:
        train_kwargs["device"] = args.device

    # DLCSPDY distillation uses Colab settings: runs/train, workers=0
    if variant == "dlcspdy":
        distill_cfg = cfg.get("distill", {})
        train_kwargs["project"] = args.project or distill_cfg.get("project", "runs/train")
        if args.workers is None:
            train_kwargs["workers"] = distill_cfg.get("workers", 0)

    return train_kwargs


def train_standard(model_path: str, data_path: str, train_kwargs: dict):
    """Standard training — same as Colab cells for YOLOv13 / DY / CSP / CSPDY."""
    model = YOLO(model_path)
    return model.train(data=data_path, **train_kwargs)


def train_dlcspdy(student_path: str, teacher_path: str, data_path: str, train_kwargs: dict):
    """DLCSPDY with knowledge distillation — same as DLPPE2025 / PPECHVGlai notebooks."""
    teacher = YOLO(teacher_path)
    head = teacher.model.model[-1]
    if hasattr(head, "set_Distillation"):
        head.set_Distillation = True

    student = YOLO(student_path)
    train_kwargs["model_t"] = teacher.model
    return student.train(data=data_path, **train_kwargs)


def parse_args() -> tuple[argparse.Namespace, dict]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    pre_args, remaining = pre.parse_known_args()
    cfg = load_config(pre_args.cfg)

    parser = argparse.ArgumentParser(
        description="Train DLCSPDY (Colab-compatible)",
        parents=[pre],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default=cfg.get("variant", "dlcspdy"),
        help="Model variant: yolov13 | dy | csp | cspdy | dlcspdy",
    )
    parser.add_argument("--model", default=None, help="Override model YAML or .pt")
    parser.add_argument("--data", default=None, help="Dataset YAML (overrides --dataset)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--teacher", default=None, help="Teacher .pt for DLCSPDY distillation")
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 200))
    parser.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 640))
    parser.add_argument("--batch", type=int, default=cfg.get("batch", 32))
    parser.add_argument("--device", default=None, help="GPU id, e.g. 0")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=cfg.get("name", "train"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args(remaining)
    return args, load_config(args.cfg)


def main() -> None:
    args, cfg = parse_args()
    variant = args.variant
    data_path = resolve_data(args, cfg)
    model_path = resolve_variant_model(variant, cfg, args.model)
    train_kwargs = build_train_kwargs(cfg, args, variant)

    print("=" * 55)
    print("DLCSPDY Training")
    print("=" * 55)
    print(f"  Variant : {variant}")
    print(f"  Config  : {args.cfg}")
    print(f"  Model   : {model_path}")
    print(f"  Data    : {data_path}")
    print(f"  Epochs  : {train_kwargs['epochs']}")
    print(f"  Batch   : {train_kwargs['batch']}")
    print(f"  ImgSize : {train_kwargs['imgsz']}")
    print(f"  Output  : {train_kwargs['project']}/{train_kwargs['name']}")

    if variant == "dlcspdy":
        teacher_path = resolve_teacher(args, cfg)
        if not teacher_path:
            raise SystemExit(
                "DLCSPDY requires a teacher checkpoint.\n"
                "  python train.py --variant dlcspdy --dataset ppe2025 "
                "--teacher runs/detect/PPEyolov13n/weights/best.pt"
            )
        print(f"  Teacher : {teacher_path}")
        print("=" * 55)
        train_dlcspdy(model_path, teacher_path, data_path, train_kwargs)
    else:
        print("=" * 55)
        train_standard(model_path, data_path, train_kwargs)


if __name__ == "__main__":
    main()
