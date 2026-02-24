import numpy as np

from core.metrics import compute_metrics


def _sample_images(seed: int = 42):
    rng = np.random.default_rng(seed)
    content = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    style = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return content, style


def test_compute_metrics_returns_expected_schema_and_range():
    content, style = _sample_images()
    output = content.copy()

    result = compute_metrics(content, style, output)

    assert set(result.keys()) == {"scores", "overall", "triple_compare", "style_match"}

    scores = result["scores"]
    assert set(scores.keys()) == {
        "content_consistency",
        "style_alignment",
        "texture_coherence",
        "structure_stability",
        "detail_fidelity",
        "color_balance",
    }
    for value in scores.values():
        assert 0.0 <= value <= 100.0
    assert 0.0 <= result["overall"] <= 100.0

    triple_compare = result["triple_compare"]
    assert set(triple_compare.keys()) == {
        "brightness_mean",
        "contrast_std",
        "saturation_mean",
        "sharpness",
        "entropy",
    }
    for payload in triple_compare.values():
        assert set(payload.keys()) == {"content", "style", "output"}
        assert payload["content"] >= 0.0
        assert payload["style"] >= 0.0
        assert payload["output"] >= 0.0

    style_match = result["style_match"]
    assert set(style_match.keys()) == {
        "color_distribution_match",
        "saturation_match",
        "texture_match",
        "entropy_match",
    }
    for value in style_match.values():
        assert 0.0 <= value <= 100.0


def test_content_consistency_improves_when_output_matches_content():
    content, style = _sample_images(7)
    noisy_output = (content.astype(np.int16) + 40).clip(0, 255).astype(np.uint8)
    same_output = content.copy()

    noisy_score = compute_metrics(content, style, noisy_output)["scores"]["content_consistency"]
    same_score = compute_metrics(content, style, same_output)["scores"]["content_consistency"]

    assert same_score >= noisy_score


def test_triple_metrics_match_when_output_equals_content():
    content, style = _sample_images(11)
    result = compute_metrics(content, style, content.copy())
    triple_compare = result["triple_compare"]
    for metric_values in triple_compare.values():
        assert metric_values["output"] == metric_values["content"]
