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

## 6. Validation
- Verified RGB-Depth file correspondence.
- Checked timestamp synchronization and duplicated indices.
- Calculated depth valid-pixel ratios.
- Recorded scene-level statistics and fog severity.
- Tested dataset loading and compatibility with Identity, CLAHE, and DCP.

## 7. Limitations
- Ground-truth detection and tracking labels are not provided.
- RGB-Depth spatial alignment and absolute distance accuracy are not independently verified.
- Detection metrics such as mAP require ground-truth annotations.
