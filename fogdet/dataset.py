
"""Gemini RGB-D dataset loader for the fogdet project.

The loader reads scenario metadata once, and only loads images on demand.
Images are returned as OpenCV BGR uint8 arrays, suitable for
Identity/CLAHE/DCP/LearnedDefog's common interface.

Usage::

    from dataset import GeminiDataset
    data = GeminiDataset()  # finds frames.csv under /kaggle/input
    bgr, depth, meta = data.load('S01', 1)
    for bgr, meta in data.iter_frames('S01', include_depth=False):
        ...
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


class GeminiDataset:
    """Read paired RGB/Depth data from S01–S07 without loading all frames into RAM.

    Parameters
    ----------
    root:
        Directory to search for scene-level ``frames.csv`` files.
        Kaggle's ``/kaggle/input`` is used by default. May also point
        directly to the dataset directory containing S01... folders.

    Notes
    -----
    ``frame_number`` is the 1-based *row position* in ``frames.csv``.
    The returned metadata contains the dataset's original ``frame_id``.
    """

    def __init__(self, root: PathLike = "/kaggle/input") -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.scenes: dict[str, dict] = {}
        for csv_path in sorted(self.root.rglob("frames.csv")):
            # Only accept scene-level CSVs with IDs such as S01, S02, etc.
            scene_dir = csv_path.parent
            scene_id = scene_dir.name.split("_", 1)[0].upper()
            if not (len(scene_id) == 3 and scene_id[0] == "S" and scene_id[1:].isdigit()):
                continue
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not {"rgb_path", "depth_path"}.issubset(reader.fieldnames):
                    raise ValueError(f"Missing rgb_path/depth_path in {csv_path}")
                rows = list(reader)
            if not rows:
                raise ValueError(f"Empty frames.csv: {csv_path}")
            if scene_id in self.scenes:
                raise ValueError(
                    f"Duplicate scene {scene_id}: {self.scenes[scene_id]['path']} and {scene_dir}. "
                    "Specify a more precise root."
                )
            self.scenes[scene_id] = {"path": scene_dir, "rows": rows}

        if not self.scenes:
            raise FileNotFoundError(f"No scenario frames.csv found under {self.root}")

    @property
    def scene_ids(self) -> list[str]:
        """Sorted IDs, e.g. ['S01', ..., 'S07']."""
        return sorted(self.scenes)

    def __len__(self) -> int:
        """Total paired RGB-D frame count across scenes."""
        return sum(len(info["rows"]) for info in self.scenes.values())

    def frame_count(self, scene_id: str) -> int:
        """Number of paired frames in a scene."""
        return len(self._scene(scene_id)["rows"])

    def _scene(self, scene_id: str) -> dict:
        key = scene_id.upper()
        if key not in self.scenes:
            raise KeyError(f"Unknown scene {scene_id!r}; available: {self.scene_ids}")
        return self.scenes[key]

    def _row_and_dir(self, scene_id: str, frame_number: int) -> tuple[dict, Path]:
        scene = self._scene(scene_id)
        if not isinstance(frame_number, int) or not 1 <= frame_number <= len(scene["rows"]):
            raise IndexError(
                f"Invalid frame_number {frame_number!r} for {scene_id}; "
                f"expected 1..{len(scene['rows'])}"
            )
        return scene["rows"][frame_number - 1], scene["path"]

    def load(
        self, scene_id: str, frame_number: int, *, include_depth: bool = True
    ) -> tuple:
        """Load a numbered paired frame (1-based row position).

        When include_depth=True: returns ``(bgr, depth_m, metadata)``.
        When include_depth=False: returns ``(bgr, metadata)``.
        BGR: uint8 HxWx3; depth: float32 HxW, meters, zero invalid.
        """
        row, scene_dir = self._row_and_dir(scene_id, frame_number)
        rgb_path = scene_dir / row["rgb_path"]
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Unable to read RGB image: {rgb_path}")
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError(f"RGB image has unexpected shape/type: {rgb_path}")

        metadata = dict(row)
        metadata.update({
            "scene_id": scene_id.upper(),
            "frame_number": frame_number,
            "rgb_file": str(rgb_path),
        })
        if not include_depth:
            return bgr, metadata

        depth_path = scene_dir / row["depth_path"]
        try:
            depth_m = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise OSError(f"Unable to read Depth array: {depth_path}") from exc
        if depth_m.shape != bgr.shape[:2] or depth_m.dtype != np.float32:
            raise ValueError(
                f"Depth shape/type mismatch: {depth_path}; "
                f"got {depth_m.shape}/{depth_m.dtype}, expected {bgr.shape[:2]}/float32"
            )
        metadata["depth_file"] = str(depth_path)
        return bgr, depth_m, metadata

    def iter_frames(
        self, scene_id: Optional[str] = None, *, include_depth: bool = False
    ) -> Iterator[tuple]:
        """Yield frames lazily in ascending scene and frame order.

        If scene_id is None, iterates over all scenes. RGB-only iteration
        is the default because dehazing takes BGR images as input.
        """
        ids = [scene_id.upper()] if scene_id is not None else self.scene_ids
        for sid in ids:
            for n in range(1, self.frame_count(sid) + 1):
                yield self.load(sid, n, include_depth=include_depth)


if __name__ == "__main__":
    dataset = GeminiDataset()
    print("Scene IDs:", dataset.scene_ids)
    print("Total paired frames:", len(dataset))
    for sid in dataset.scene_ids:
        print(f"  {sid}: {dataset.frame_count(sid)} pairs")
    bgr, depth, meta = dataset.load(dataset.scene_ids[0], 1)
    print("Sample BGR:", bgr.shape, bgr.dtype)
    print("Sample Depth:", depth.shape, depth.dtype)
    print("Sample metadata frame_id:", meta.get("frame_id"))
