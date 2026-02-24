import numpy as np
from core.transfer import reinhard_color_transfer


def test_reinhard_color_transfer_shape_and_range():
    content = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    style = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    out = reinhard_color_transfer(content, style, strength=1.0)
    assert out.shape == content.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
