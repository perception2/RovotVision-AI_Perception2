# CLAHE + DCP (dehazing, Stage 1) — Robin Senechal

Two classical dehazing methods, implemented with one shared interface so every method of the project (identity, CLAHE, DCP, learned) plugs into the same detection and evaluation pipeline.

## Interface (for every dehazing method)

```python
out = method(frame)
# frame, out: numpy array (H, W, 3), uint8, BGR (OpenCV order), same size
```

To add a method, subclass `DefogBase` in `defog/clahe_dcp.py` (or follow the same contract) and register it in `REGISTRY`.

## Methods

| Method | What it does | Parameters used |
| --- | --- | --- |
| `identity` | No processing. Baseline, never removed. | — |
| `clahe` | CLAHE on the L channel of LAB: local contrast boost, no physical model. | clipLimit 2.0, 8×8 tiles |
| `dcp` | Dark Channel Prior (He et al., CVPR 2009): estimates airlight and transmission from the dark channel, refines transmission with a guided filter (He et al., ECCV 2010), restores J = (I − A) / max(t, t0) + A. | patch 15, ω 0.95, t0 0.1, guided filter on |

## How to run

Requirements: Python 3.11, `opencv-python`, `numpy`.

```bash
pip install opencv-python numpy

# apply both methods to a folder of images
python defog/clahe_dcp.py --input path/to/rgb --output out/ --method both
```

In Python:

```python
import cv2
from defog.clahe_dcp import build_defog

img = cv2.imread("000001.png")          # BGR uint8
clahe = build_defog("clahe")
dcp = build_defog("dcp")
cv2.imwrite("clahe.png", clahe(img))
cv2.imwrite("dcp.png", dcp(img))
```

On the TA dataset (Kaggle): `notebooks/fogdet_clahe_dcp_ta_dataset.ipynb`. Add the Kaggle dataset `perception_fog_inference_dataset` as input, turn Internet on (for YOLO weights), Run All. Outputs: side-by-side videos with YOLO boxes, before/after figures, `results.md`.

## Results

### RTTS (real fog, labelled) — detection

YOLOv8n (COCO), conf 0.25, IoU 0.7, imgsz 640, 5 RTTS classes, 600 images, 3 runs (identity/CLAHE/DCP are deterministic, std 0). Clear-weather check on a COCO val2017 subset.

| Condition | mAP@50 | Δ mAP@50 (pp) | mAP@50-95 | Δ clear-weather mAP@50 (pp) | PSNR | SSIM | FPS (CPU, defog + YOLO) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Identity | 0.341 | — | 0.222 | — | 13.60 (hazy) | 0.740 (hazy) | 12.4 |
| CLAHE | 0.370 | +2.90 | 0.235 | −1.22 | 14.50 | 0.753 | 5.1 |
| DCP | 0.368 | +2.69 | 0.236 | +0.58 | 18.70 | 0.857 | 1.6 |

PSNR/SSIM on RESIDE-6K test pairs.

### TA dataset (Gemini 345Lg, S01–S07, 7,028 frames) — no labels

The TA dataset has no detection labels, so mAP cannot be computed (guide 1-2, option 2: mAP on a public dataset + qualitative comparison on the provided data). Label-free image metrics, mean ± std over frames:

| Condition | Fog score (dark channel) ↓ | RMS contrast ↑ | Sharpness (Laplacian var.) ↑ | FPS (CPU, defog only) |
| --- | --- | --- | --- | --- |
| Identity | 0.508 ± 0.046 | 0.066 ± 0.017 | 144 ± 67 | — |
| CLAHE | 0.433 ± 0.062 | 0.114 ± 0.024 | 654 ± 300 | 117 |
| DCP | 0.100 ± 0.023 | 0.120 ± 0.021 | 792 ± 370 | 7.2 |

## Findings and limits

- On real fog, both methods raise mAP@50 by about 2.7–2.9 points. DCP does not hurt clear-weather detection; CLAHE costs 1.2 points.
- On the TA data, both roughly double contrast; DCP gives the best image in 6 of 7 scenarios (CLAHE slightly ahead on S02/S03).
- The fog score favours DCP by construction (DCP minimises the dark channel); contrast and sharpness are the fair comparison.
- YOLO detections on the TA data are not meaningful: the objects (packed meat, icebox) are not COCO classes (meat is detected as "cat", "cake", "refrigerator"; the icebox as "toilet", "suitcase"). Failure cases for the report.
- Only CLAHE runs in real time on CPU (117 FPS vs 7 FPS for DCP).

## References

He, Sun, Tang, *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009 · He, Sun, Tang, *Guided Image Filtering*, ECCV 2010 · Zuiderveld, *Contrast Limited Adaptive Histogram Equalization*, Graphics Gems IV, 1994 · Li et al., *Benchmarking Single-Image Dehazing and Beyond* (RESIDE / RTTS), TIP 2019.
