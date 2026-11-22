# Gemini RGB-D Dataset Loader

## 1. Overview
Common RGB-D dataset loader for the Fog Detection project.

- Dataset: Gemini 345Lg
- Scenes: S01–S07 (7 scenes)
- Paired RGB-Depth frames: 7,028
- RGB: 1280×720, uint8, BGR
- Depth: 1280×720, float32, meters

## 2. Environment
- Platform: Kaggle Notebook
- Python 3
- Dependencies: NumPy, OpenCV

## 3. Dataset Preparation
Attach the Gemini RGB-D dataset to the Kaggle notebook.

The loader searches for scene-level frames.csv files under /kaggle/input.

## 4. Installation
Install dependencies if needed:

```bash
pip install numpy opencv-python
```

## 5. Usage
```python
from fogdet.dataset import GeminiDataset

dataset = GeminiDataset()
print(dataset.scene_ids)
print(len(dataset))

rgb, depth, meta = dataset.load("S01", 1)

print(rgb.shape, rgb.dtype)
print(depth.shape, depth.dtype)
print(meta["frame_id"])
```

Expected results:
- 7 scenes
- 7,028 paired frames
- RGB shape: (720, 1280, 3), uint8
- Depth shape: (720, 1280), float32


## 6. Dataset Validation

### 6.1 Validation Scope

The Gemini dataset was inspected using the provided RGB-D frame pairs.

- Scenarios: 7 (S01–S07)
- Total synchronized RGB-Depth pairs: 7,028
- Resolution: 1280 × 720
- RGB format: PNG, uint8, BGR
- Depth format: NPY, float32, meters

The original dataset contains additional unmatched frames.
Only synchronized pairs listed in `frames.csv` are used
for the common dataset loader.

### 6.2 Validation Checks

The following checks were performed:

- Verified RGB and Depth frame numbering.
- Checked the existence of corresponding RGB/Depth files.
- Verified RGB-Depth timestamp differences within 20 ms.
- Checked duplicated source frame indices.
- Verified image shapes and data types.
- Calculated FPS using RGB sensor timestamps.
- Summarized scene-level Depth valid-pixel ratios.
- Inspected five representative RGB frames per scene.
- Classified fog severity for S01–S07.

The metadata and file consistency checks passed
for all seven scenarios.

### 6.3 Fog Severity Classification

| Scene | Fog Level |
|---|---|
| S01 | Medium |
| S02 | Light |
| S03 | Light |
| S04 | Medium |
| S05 | Heavy |
| S06 | Heavy |
| S07 | Medium |

The fog levels were assigned by visual inspection
of five sampled frames from each scene.

No clearly fog-free scene was observed in the sampled
frames. This does not establish that every frame
contains fog.

### 6.4 Output Files

Validation results are available in the `results/`
directory:

| File | Description |
|---|---|
| `scene_summary.csv` | Frame counts, FPS, duration, and Depth valid-pixel ratios |
| `data_checks.csv` | RGB-Depth consistency and metadata validation results |
| `fog_levels.csv` | Scene-level fog severity classifications |

### 6.5 Reproducing the Validation

Open the following notebook in Kaggle:

`notebooks/dataset_validation_and_loader.ipynb`

1. Attach the Gemini RGB-D dataset to the notebook.
2. Run the Dataset Validation cells in order.
3. Confirm that all seven scenes pass the checks.
4. Review the sampled RGB images.
5. Find the generated CSV files in `/kaggle/working/`.

The validation code checks metadata, file existence,
and one RGB-Depth sample pair per scene. A separate
full-frame decoding check was performed previously.

### 6.6 Remaining Work

- Finalize the evaluation annotation strategy.
- Record the scene-level dataset split in the repository.
- Integrate the loader with the final shared pipeline.


## 7. Limitations
- Ground-truth detection and tracking labels are not provided.
- RGB-Depth spatial alignment and absolute distance accuracy are not independently verified.
- Detection metrics such as mAP require ground-truth annotations.
