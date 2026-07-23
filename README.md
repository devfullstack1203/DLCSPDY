# DLCSPDY

**DLCSPDY** (Dynamic Convolution + CSP + YOLOv13) is a modified Ultralytics YOLO framework for **Personal Protective Equipment (PPE)** object detection. The proposed backbone replaces standard C3k2 blocks with `C3k2_DynamicConv` modules (see `ultralytics/nn/newModule/DyConv.py`).

## Repository structure

```
DLCSPDY/
├── configs/                  # Dataset & training configs for reproduction
│   ├── datasets/             # PPE2025, CHVG, CPPE
│   └── train_default.yaml    # Default hyperparameters
├── datasets/                 # YOLO-format datasets (see datasets/README.md)
├── ultralytics/              # Modified Ultralytics source
│   └── cfg/models/v13/
│       └── CSPDYyolov13.yaml # Proposed model architecture
├── train.py                  # Training entry point
├── val.py                    # Evaluation entry point
├── requirements.txt          # Pinned dependencies
└── LICENSE                   # AGPL-3.0
```

## Requirements

- Python 3.12 (experiments run on 3.12.12)
- PyTorch 2.9.0 + CUDA-capable GPU (experiments on **NVIDIA L4, 22 GB**)
- Linux recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/vttai-dev/DLCSPDY.git
cd DLCSPDY

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install this package in editable mode
pip install -e .
```

> **Note:** `flash-attn` in `requirements.txt` is optional and only needed on Linux + CUDA 11. Install the matching wheel manually if required.

## Datasets

Place or verify datasets under `datasets/`:

| Dataset  | Classes | Train | Val | Test |
|----------|---------|------:|----:|-----:|
| PPE2025  | 4       | 1829  | 228 |  228 |
| CHVG     | 8       | 1358  | 169 |  171 |
| CPPE     | 4       | 2545  | 474 |  161 |

See [datasets/README.md](datasets/README.md) for class names, licenses, and split details.

## Model architecture

The proposed model is defined in:

```
ultralytics/cfg/models/v13/CSPDYyolov13.yaml
```

Scale variants (`n`, `s`, `m`, `l`, `x`) follow Ultralytics naming, e.g. `CSPDYyolov13n.yaml`.

Baseline variants for comparison:

- `CSPyolov13.yaml` — CSP backbone (DSC3k2)
- `DYyolov13.yaml` — Dynamic Conv only
- `yolov13.yaml` — Original YOLOv13

## Training configuration

All experiments use the settings in [`configs/train_default.yaml`](configs/train_default.yaml):

| Category | Setting | Value |
|----------|---------|-------|
| **Hardware** | GPU | NVIDIA L4 (22,693 MiB) |
| | Framework | Ultralytics YOLO (PyTorch 2.9.0) |
| | Python | 3.12.12 |
| **Input** | Image size | 640 × 640 |
| | Batch size | 32 |
| | Epochs | 200 |
| | Workers | 8 |
| **Optimizer** | Type | SGD (auto-selected) |
| | Initial LR | 0.01 |
| | Momentum | 0.9 |
| | Weight decay | 5 × 10⁻⁴ |
| **Warm-up** | Epochs | 3 |
| | Momentum | 0.8 |
| **Loss weights** | Box / Cls / DFL | 7.5 / 0.5 / 1.5 |
| **Augmentation** | Mosaic | on (off last 10 epochs) |
| | Horizontal flip | 0.5 |
| | Translation / Scale | 0.1 / 0.5 |
| | HSV | (0.015, 0.7, 0.4) |
| | Copy-paste | 0.1 (flip) |
| | Auto augment | RandAugment |
| **Strategy** | AMP | enabled |
| | Deterministic | enabled |
| | Val during training | yes |
| | Checkpoint | every 50 epochs |

## Training

`train.py` mirrors the Colab notebooks (`PPE2025.ipynb`, `PPECHVGlai.ipynb`, `DLPPEENV.ipynb`).

### Step 1 — Train baselines (teacher models)

```bash
# YOLOv13 baseline (teacher for DLCSPDY)
python train.py --variant yolov13 --dataset ppe2025 --name PPEyolov13n
python train.py --variant yolov13 --dataset chvg   --name CHVGyolov13nLAI
python train.py --variant yolov13 --dataset cppe   --name ENVyolov13n

# Other baselines for ablation
python train.py --variant dy    --dataset ppe2025 --name PPEDYyolo
python train.py --variant csp   --dataset ppe2025 --name PPECSPyolo
python train.py --variant cspdy --dataset ppe2025 --name PPECSPDYyolo
```

### Step 2 — Train DLCSPDY (knowledge distillation)

```bash
python train.py --variant dlcspdy --dataset ppe2025 \
  --teacher runs/detect/PPEyolov13n/weights/best.pt \
  --name PPEDLCSPDYyolo

python train.py --variant dlcspdy --dataset chvg \
  --teacher runs/detect/CHVGyolov13nLAI/weights/best.pt \
  --name CHVGDLCSPDYyolo

python train.py --variant dlcspdy --dataset cppe \
  --teacher runs/detect/ENVyolov13n/weights/best.pt \
  --name ENVDLCSPDYyolo
```

DLCSPDY uses `project=runs/train`, `workers=0`, and `model_t` (teacher) — same as Colab.

### Option 2 — Ultralytics CLI

```bash
yolo detect train \
  model=ultralytics/cfg/models/v13/CSPDYyolov13n.yaml \
  data=configs/datasets/ppe2025.yaml \
  epochs=200 imgsz=640 batch=32 save_period=50
```

Training outputs are saved to `runs/detect/<name>/` (weights, metrics, plots).

## Evaluation

### Option 1 — `val.py` (recommended)

```bash
# Validate on val split (default)
python val.py --weights runs/detect/train/weights/best.pt --data configs/datasets/ppe2025.yaml

# Evaluate on test split
python val.py --weights runs/detect/train/weights/best.pt --data configs/datasets/ppe2025.yaml --split test
```

### Option 2 — Ultralytics CLI

```bash
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=configs/datasets/ppe2025.yaml \
  split=test
```

## Inference

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=path/to/images \
  conf=0.25
```

## Reproducibility checklist

| Item | Location |
|------|----------|
| Install guide | This README |
| Dependencies | `requirements.txt` |
| Model architecture | `ultralytics/cfg/models/v13/CSPDYyolov13.yaml` |
| Training config | `configs/train_default.yaml` |
| Dataset configs | `configs/datasets/*.yaml` |
| Training script | `train.py` |
| Evaluation script | `val.py` |
| Dataset documentation | `datasets/README.md` |
| License | `LICENSE` (AGPL-3.0) |

## License

This project is based on [Ultralytics](https://github.com/ultralytics/ultralytics) and is released under the **GNU Affero General Public License v3.0** (see [LICENSE](LICENSE)).

Dataset licenses differ per source — see [datasets/README.md](datasets/README.md).

## Citation

If you use this code or datasets in your research, please cite the corresponding paper and acknowledge the Ultralytics YOLO framework.
