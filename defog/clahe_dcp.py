"""CLAHE and Dark Channel Prior (DCP) dehazing. Robot Vision & AI, Perception team 2.

Part: CLAHE + DCP (Robin Senechal).

Common interface for every dehazing method of the project:
    out = method(frame)
    frame, out: (H, W, 3) uint8 BGR (OpenCV order), same size.
Any new method (e.g. the learned one) only has to follow this contract to plug into the
same detection / evaluation pipeline.

Usage as a script (process a folder of images):
    python defog/clahe_dcp.py --input path/to/rgb --output out/ --method both
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class DefogBase:
    """Both input and output are (H, W, 3) uint8 BGR. Do not change this signature."""

    name = "base"

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ============================================================================ identity
class IdentityDefog(DefogBase):
    """Baseline that performs no processing. Never remove it."""

    name = "identity"

    def __call__(self, frame):
        return frame


# ============================================================================ CLAHE
class ClaheDefog(DefogBase):
    """CLAHE from W1. Locally increases contrast only on the luminance channel."""

    name = "clahe"

    def __init__(self, clip=2.0, grid=8):
        self.clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))

    def __call__(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================================ DCP
def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Gray-guide guided filter (He et al., ECCV 2010), O(N) with box filters.

    guide, src: float32 (H, W) in [0, 1]. Edge-preserving smoothing of `src` following
    the edges of `guide` — removes the blocky halos of the patch-wise transmission.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide)
    mean_p = box(src)
    corr_ip = box(guide * src)
    corr_ii = box(guide * guide)
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


class DcpDefog(DefogBase):
    """Dark Channel Prior (He et al., CVPR 2009). No training is required.

    1) Estimate atmospheric light A from the brightest top 0.1% pixels of the dark channel.
    2) Estimate transmission: t = 1 - omega * dark(I / A)
    3) Refine t with a guided filter and restore J = (I - A) / max(t, t0) + A.

    Args:
        patch:      dark-channel patch size (15 in the paper).
        omega:      keeps a little haze for depth perception (0.95 in the paper).
        t0:         lower bound on t, avoids noise amplification in dense fog.
        top:        fraction of the brightest dark-channel pixels used to estimate A.
        guided:     refine t with the guided filter (set False for the halo ablation).
        gf_radius:  guided-filter radius in pixels; None -> scales with image size.
        gf_eps:     guided-filter regularisation.
        a_max:      clip on A to avoid over-darkening when sky/white walls dominate.
    """

    name = "dcp"

    def __init__(self, patch=15, omega=0.95, t0=0.1, top=0.001, guided=True,
                 gf_radius=None, gf_eps=1e-3, a_max=0.98):
        self.patch, self.omega, self.t0, self.top = int(patch), float(omega), float(t0), float(top)
        self.guided, self.gf_radius, self.gf_eps, self.a_max = bool(guided), gf_radius, float(gf_eps), float(a_max)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch, self.patch))

    def dark_channel(self, img: np.ndarray) -> np.ndarray:
        """min over RGB, then min filter (erosion) over a patch. Works on uint8 or float."""
        return cv2.erode(img.min(axis=2), self._kernel)

    def estimate_atmospheric_light(self, img: np.ndarray, dark: np.ndarray) -> np.ndarray:
        h, w = dark.shape
        n = max(1, int(h * w * self.top))
        idx = np.argpartition(dark.ravel(), -n)[-n:]
        candidates = img.reshape(-1, 3)[idx]
        # He et al. pick the single brightest candidate; the mean of the candidates is
        # more robust to isolated specular pixels and gives nearly the same A.
        A = candidates.mean(axis=0)
        return np.clip(A, 1e-3, self.a_max).astype(np.float32)

    def restore(self, frame: np.ndarray) -> dict:
        """Full DCP with intermediates (used for figures and the guided-filter ablation)."""
        img = frame.astype(np.float32) / 255.0
        dark = self.dark_channel(img)
        A = self.estimate_atmospheric_light(img, dark)
        t_raw = 1.0 - self.omega * self.dark_channel(img / A[None, None, :])
        if self.guided:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            r = self.gf_radius or max(8, int(round(min(gray.shape) / 18)))  # ~60 px at 1080p
            t = guided_filter(gray, t_raw.astype(np.float32), int(r), self.gf_eps)
        else:
            t = t_raw
        t = np.clip(t, self.t0, 1.0)
        J = (img - A[None, None, :]) / t[..., None] + A[None, None, :]
        out = np.clip(J * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return {"dark": dark, "A": A, "t_raw": t_raw, "t": t, "out": out}

    def __call__(self, frame):
        return self.restore(frame)["out"]


REGISTRY = {"identity": IdentityDefog, "clahe": ClaheDefog, "dcp": DcpDefog}


def build_defog(name: str, **kwargs) -> DefogBase:
    """build_defog("clahe") / build_defog("dcp", omega=0.9) ..."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Choose from {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


# ------------------------------------------------------------------ unicode-safe image I/O (Windows)
def imread(path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        buf.tofile(str(path))


def main():
    ap = argparse.ArgumentParser(description="Apply CLAHE and/or DCP to a folder of images.")
    ap.add_argument("--input", required=True, help="image file or folder (png/jpg)")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--method", choices=["clahe", "dcp", "both"], default="both")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.max_frames:
        files = files[: args.max_frames]
    methods = ["clahe", "dcp"] if args.method == "both" else [args.method]
    defogs = {m: build_defog(m) for m in methods}
    times = {m: [] for m in methods}
    for f in files:
        img = imread(f)
        if img is None:
            continue
        for m, fn in defogs.items():
            t0 = time.perf_counter()
            out = fn(img)
            times[m].append((time.perf_counter() - t0) * 1000)
            imwrite(Path(args.output) / m / f.relative_to(src if src.is_dir() else src.parent).with_suffix(".png"), out)
    for m, t in times.items():
        if t:
            print(f"{m}: {len(t)} frames, {np.mean(t):.1f} ms/frame ({1000 / np.mean(t):.1f} FPS)")


if __name__ == "__main__":
    main()
