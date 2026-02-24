from __future__ import annotations

import cv2
import numpy as np


LAB_MIN = np.array([0, 0, 0], dtype=np.float32)
LAB_MAX = np.array([255, 255, 255], dtype=np.float32)


def _lab_stats(img_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = img_lab.reshape(-1, 3).mean(axis=0)
    std = img_lab.reshape(-1, 3).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def reinhard_color_transfer(content_bgr: np.ndarray, style_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    content_lab = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    style_lab = cv2.cvtColor(style_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    c_mean, c_std = _lab_stats(content_lab)
    s_mean, s_std = _lab_stats(style_lab)

    transferred = (content_lab - c_mean) * (s_std / c_std) + s_mean
    transferred = np.clip(transferred, LAB_MIN, LAB_MAX)

    blended = content_lab * (1.0 - strength) + transferred * strength
    blended = np.clip(blended, LAB_MIN, LAB_MAX).astype(np.uint8)

    return cv2.cvtColor(blended, cv2.COLOR_LAB2BGR)


def apply_bilateral(img_bgr: np.ndarray, edge_strength: float) -> np.ndarray:
    edge_strength = float(np.clip(edge_strength, 0.0, 1.0))
    # Map strength to reasonable bilateral parameters
    sigma_color = 10 + edge_strength * 60
    sigma_space = 5 + edge_strength * 20
    return cv2.bilateralFilter(img_bgr, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def detail_boost(content_bgr: np.ndarray, base_bgr: np.ndarray, boost: float) -> np.ndarray:
    boost = float(np.clip(boost, 0.0, 1.0))
    blur = cv2.GaussianBlur(content_bgr, (0, 0), sigmaX=3.0)
    detail = content_bgr.astype(np.float32) - blur.astype(np.float32)
    out = base_bgr.astype(np.float32) + detail * (boost * 1.5)
    return np.clip(out, 0, 255).astype(np.uint8)


def transfer_style(
    content_bgr: np.ndarray,
    style_bgr: np.ndarray,
    color_strength: float,
    edge_strength: float,
    detail_strength: float,
) -> np.ndarray:
    colored = reinhard_color_transfer(content_bgr, style_bgr, strength=color_strength)
    smoothed = apply_bilateral(colored, edge_strength=edge_strength)
    out = detail_boost(content_bgr, smoothed, boost=detail_strength)
    return out
