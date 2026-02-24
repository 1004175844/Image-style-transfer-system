from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageTk


PREVIEW_SIZE = (300, 300)


def pil_to_tk(img: Image.Image) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(img)


def fit_image(img: Image.Image, size: Tuple[int, int] = PREVIEW_SIZE) -> Image.Image:
    img = img.copy()
    img.thumbnail(size, Image.LANCZOS)
    return img
