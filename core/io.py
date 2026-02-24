from __future__ import annotations

import os

import cv2
import numpy as np


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTS


def load_image_bgr(path: str) -> np.ndarray:
    # Use imdecode + fromfile for Windows Unicode path compatibility.
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Failed to load image: {path} (empty file or unreadable path)")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_image_bgr(path: str, img: np.ndarray) -> None:
    # Use imencode + tofile for Windows Unicode path compatibility.
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        raise ValueError(f"Failed to save image: {path} (output folder not found)")

    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Failed to save image: {path} (unsupported extension: {ext or 'none'})")

    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to save image: {path} (encode failed)")

    try:
        encoded.tofile(path)
    except Exception as exc:
        raise ValueError(f"Failed to save image: {path} ({exc})") from exc


def resize_to_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # Resize src to match ref size by scaling and center-cropping.
    ref_h, ref_w = ref.shape[:2]
    src_h, src_w = src.shape[:2]

    scale = max(ref_w / src_w, ref_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop
    x0 = (new_w - ref_w) // 2
    y0 = (new_h - ref_h) // 2
    return resized[y0:y0 + ref_h, x0:x0 + ref_w]
