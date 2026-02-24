from __future__ import annotations

import cv2
import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _to_score(value01: float) -> float:
    return _clip01(value01) * 100.0


def _ssim_gray(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)

    c1 = 6.5025
    c2 = 58.5225

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    num = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = num / (den + 1e-8)
    return float(np.mean(ssim_map))


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float32)
    bb = b.reshape(-1).astype(np.float32)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if float(np.std(aa)) < 1e-6 or float(np.std(bb)) < 1e-6:
        return 0.0
    corr = float(np.corrcoef(aa, bb)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def _hist_similarity(style_bgr: np.ndarray, output_bgr: np.ndarray) -> float:
    style_hsv = cv2.cvtColor(style_bgr, cv2.COLOR_BGR2HSV)
    out_hsv = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2HSV)
    hist_style = cv2.calcHist([style_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_out = cv2.calcHist([out_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_style, hist_style)
    cv2.normalize(hist_out, hist_out)
    corr = cv2.compareHist(hist_style, hist_out, cv2.HISTCMP_CORREL)
    return float((corr + 1.0) * 0.5)


def _closeness_ratio(a: float, b: float) -> float:
    ma = max(abs(a), 1e-6)
    mb = max(abs(b), 1e-6)
    ratio = min(ma, mb) / max(ma, mb)
    return float(np.clip(ratio, 0.0, 1.0))


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _gray_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    total = float(hist.sum())
    if total < 1e-6:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mean_saturation(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[..., 1]))


def _sharpness_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(lap))


def compute_metrics(content_bgr: np.ndarray, style_bgr: np.ndarray, output_bgr: np.ndarray) -> dict[str, object]:
    content_gray = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2GRAY)
    style_gray = cv2.cvtColor(style_bgr, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)

    content_consistency = _to_score((_ssim_gray(content_gray, output_gray) + 1.0) * 0.5)
    style_alignment = _to_score(_hist_similarity(style_bgr, output_bgr))

    lap_style = cv2.Laplacian(style_gray, cv2.CV_32F)
    lap_out = cv2.Laplacian(output_gray, cv2.CV_32F)
    texture_coherence = _to_score(_closeness_ratio(float(np.std(lap_style)), float(np.std(lap_out))))

    grad_content = _gradient_magnitude(content_gray)
    grad_out = _gradient_magnitude(output_gray)
    structure_stability = _to_score((_corrcoef(grad_content, grad_out) + 1.0) * 0.5)

    hf_content = content_gray.astype(np.float32) - cv2.GaussianBlur(content_gray.astype(np.float32), (0, 0), 2.0)
    hf_out = output_gray.astype(np.float32) - cv2.GaussianBlur(output_gray.astype(np.float32), (0, 0), 2.0)
    detail_fidelity = _to_score(_closeness_ratio(float(np.std(hf_content)), float(np.std(hf_out))))

    style_lab = cv2.cvtColor(style_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    out_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    mean_dist = float(np.linalg.norm(style_lab.mean(axis=0) - out_lab.mean(axis=0)))
    std_dist = float(np.linalg.norm(style_lab.std(axis=0) - out_lab.std(axis=0)))
    color_balance = _to_score(1.0 - min((mean_dist + std_dist) / 220.0, 1.0))

    scores = {
        "content_consistency": float(np.clip(content_consistency, 0.0, 100.0)),
        "style_alignment": float(np.clip(style_alignment, 0.0, 100.0)),
        "texture_coherence": float(np.clip(texture_coherence, 0.0, 100.0)),
        "structure_stability": float(np.clip(structure_stability, 0.0, 100.0)),
        "detail_fidelity": float(np.clip(detail_fidelity, 0.0, 100.0)),
        "color_balance": float(np.clip(color_balance, 0.0, 100.0)),
    }
    overall = float(np.mean(list(scores.values())))

    triple_compare = {
        "brightness_mean": {
            "content": float(np.mean(content_gray)),
            "style": float(np.mean(style_gray)),
            "output": float(np.mean(output_gray)),
        },
        "contrast_std": {
            "content": float(np.std(content_gray)),
            "style": float(np.std(style_gray)),
            "output": float(np.std(output_gray)),
        },
        "saturation_mean": {
            "content": _mean_saturation(content_bgr),
            "style": _mean_saturation(style_bgr),
            "output": _mean_saturation(output_bgr),
        },
        "sharpness": {
            "content": _sharpness_laplacian(content_gray),
            "style": _sharpness_laplacian(style_gray),
            "output": _sharpness_laplacian(output_gray),
        },
        "entropy": {
            "content": _gray_entropy(content_gray),
            "style": _gray_entropy(style_gray),
            "output": _gray_entropy(output_gray),
        },
    }

    style_match = {
        "color_distribution_match": float(np.clip(_to_score(_hist_similarity(style_bgr, output_bgr)), 0.0, 100.0)),
        "saturation_match": float(
            np.clip(
                _to_score(_closeness_ratio(_mean_saturation(style_bgr), _mean_saturation(output_bgr))),
                0.0,
                100.0,
            )
        ),
        "texture_match": float(
            np.clip(
                _to_score(_closeness_ratio(_sharpness_laplacian(style_gray), _sharpness_laplacian(output_gray))),
                0.0,
                100.0,
            )
        ),
        "entropy_match": float(
            np.clip(
                _to_score(_closeness_ratio(_gray_entropy(style_gray), _gray_entropy(output_gray))),
                0.0,
                100.0,
            )
        ),
    }

    return {
        "scores": scores,
        "overall": float(np.clip(overall, 0.0, 100.0)),
        "triple_compare": triple_compare,
        "style_match": style_match,
    }
