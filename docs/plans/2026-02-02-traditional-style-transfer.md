# Traditional Style Transfer (No-Training) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a simple Python desktop app that performs traditional, no-training style transfer using content + style images with a minimal UI.

**Architecture:** Separate UI from core processing. Core pipeline handles color transfer, edge-preserving smoothing, and detail boost. UI manages file selection, previews, and status updates while running processing in a background thread.

**Tech Stack:** Python 3, Tkinter, Pillow, OpenCV, NumPy, PyTest.

---

### Task 1: Create project skeleton and requirements

**Files:**
- Create: `requirements.txt`
- Create: `core/__init__.py`
- Create: `core/transfer.py`
- Create: `core/io.py`
- Create: `ui/__init__.py`
- Create: `ui/widgets.py`
- Create: `tests/test_transfer.py`
- Create: `main.py`

**Step 1: Write requirements file**

`requirements.txt`:
```
opencv-python
Pillow
numpy
pytest
```

**Step 2: Create empty package files**

`core/__init__.py`:
```
# Core package
```

`ui/__init__.py`:
```
# UI package
```

**Step 3: Run a quick import sanity check**

Run: `python -c "import sys"`
Expected: Exit 0

---

### Task 2: Write failing tests for core transfer functions

**Files:**
- Modify: `tests/test_transfer.py`

**Step 1: Write the failing test**

`tests/test_transfer.py`:
```python
import numpy as np
from core.transfer import reinhard_color_transfer


def test_reinhard_color_transfer_shape_and_range():
    content = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    style = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    out = reinhard_color_transfer(content, style, strength=1.0)
    assert out.shape == content.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transfer.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError" until implementation exists

---

### Task 3: Implement IO helpers for load/resize/save

**Files:**
- Modify: `core/io.py`

**Step 1: Write minimal IO helpers**

`core/io.py`:
```python
from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTS


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_image_bgr(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise ValueError(f"Failed to save image: {path}")


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
```

**Step 2: Run tests (still expected to fail)**

Run: `python -m pytest tests/test_transfer.py -v`
Expected: FAIL (transfer functions still missing)

---

### Task 4: Implement transfer pipeline (Reinhard + bilateral + detail boost)

**Files:**
- Modify: `core/transfer.py`

**Step 1: Implement reinhard color transfer and pipeline**

`core/transfer.py`:
```python
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


def transfer_style(content_bgr: np.ndarray, style_bgr: np.ndarray, color_strength: float, edge_strength: float, detail_strength: float) -> np.ndarray:
    colored = reinhard_color_transfer(content_bgr, style_bgr, strength=color_strength)
    smoothed = apply_bilateral(colored, edge_strength=edge_strength)
    out = detail_boost(content_bgr, smoothed, boost=detail_strength)
    return out
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_transfer.py -v`
Expected: PASS

---

### Task 5: Implement UI preview helpers

**Files:**
- Modify: `ui/widgets.py`

**Step 1: Add image-to-Tk conversion helper**

`ui/widgets.py`:
```python
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
```

---

### Task 6: Build main UI and wire events

**Files:**
- Modify: `main.py`

**Step 1: Create Tkinter window layout**

`main.py` (initial skeleton):
```python
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2
from PIL import Image

from core.io import is_image_file, load_image_bgr, resize_to_match, save_image_bgr
from core.transfer import transfer_style
from ui.widgets import fit_image, pil_to_tk


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Traditional Style Transfer")
        self.geometry("1100x700")

        self.input_dir = tk.StringVar()
        self.style_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status_text = tk.StringVar(value="Idle")

        self.color_strength = tk.DoubleVar(value=1.0)
        self.edge_strength = tk.DoubleVar(value=0.5)
        self.detail_strength = tk.DoubleVar(value=0.4)

        self.content_files: list[str] = []
        self.content_path: Optional[str] = None
        self.output_path: Optional[str] = None

        self._content_img_tk = None
        self._style_img_tk = None
        self._output_img_tk = None

        self._build_ui()

    def _build_ui(self) -> None:
        # Top controls
        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(ctrl, text="Input Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.input_dir, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(ctrl, text="Browse", command=self._pick_input_dir).grid(row=0, column=2, padx=5)

        ttk.Label(ctrl, text="Content Image").grid(row=1, column=0, sticky="w")
        self.content_combo = ttk.Combobox(ctrl, width=57, state="readonly")
        self.content_combo.grid(row=1, column=1, padx=5)
        self.content_combo.bind("<<ComboboxSelected>>", self._on_content_selected)

        ttk.Label(ctrl, text="Style Image").grid(row=2, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.style_path, width=60).grid(row=2, column=1, padx=5)
        ttk.Button(ctrl, text="Browse", command=self._pick_style_file).grid(row=2, column=2, padx=5)

        ttk.Label(ctrl, text="Output Folder").grid(row=3, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.output_dir, width=60).grid(row=3, column=1, padx=5)
        ttk.Button(ctrl, text="Browse", command=self._pick_output_dir).grid(row=3, column=2, padx=5)

        # Previews
        previews = ttk.Frame(self)
        previews.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.content_label = ttk.Label(previews, text="Content", anchor="center")
        self.style_label = ttk.Label(previews, text="Style", anchor="center")
        self.output_label = ttk.Label(previews, text="Output", anchor="center")

        self.content_label.grid(row=0, column=0, padx=10)
        self.style_label.grid(row=0, column=1, padx=10)
        self.output_label.grid(row=0, column=2, padx=10)

        # Sliders + Run
        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(controls, text="Color Strength").grid(row=0, column=0, sticky="w")
        ttk.Scale(controls, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.color_strength).grid(row=0, column=1, sticky="ew", padx=5)

        ttk.Label(controls, text="Edge Strength").grid(row=1, column=0, sticky="w")
        ttk.Scale(controls, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.edge_strength).grid(row=1, column=1, sticky="ew", padx=5)

        ttk.Label(controls, text="Detail Boost").grid(row=2, column=0, sticky="w")
        ttk.Scale(controls, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.detail_strength).grid(row=2, column=1, sticky="ew", padx=5)

        ttk.Button(controls, text="Run", command=self._run_transfer).grid(row=0, column=2, rowspan=3, padx=10)
        controls.columnconfigure(1, weight=1)

        # Status bar
        status = ttk.Label(self, textvariable=self.status_text, relief=tk.SUNKEN, anchor="w")
        status.pack(fill=tk.X, padx=2, pady=2)

    def _pick_input_dir(self) -> None:
        path = filedialog.askdirectory()
        if not path:
            return
        self.input_dir.set(path)
        self._load_content_list(path)

    def _load_content_list(self, path: str) -> None:
        files = [f for f in os.listdir(path) if is_image_file(f)]
        files.sort()
        self.content_files = [os.path.join(path, f) for f in files]
        self.content_combo["values"] = files
        if files:
            self.content_combo.current(0)
            self._on_content_selected()

    def _on_content_selected(self, _evt=None) -> None:
        idx = self.content_combo.current()
        if idx < 0:
            return
        self.content_path = self.content_files[idx]
        self._update_preview(self.content_path, target="content")

    def _pick_style_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        self.style_path.set(path)
        self._update_preview(path, target="style")

    def _pick_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if not path:
            return
        self.output_dir.set(path)

    def _update_preview(self, img_path: str, target: str) -> None:
        img_bgr = load_image_bgr(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        pil = fit_image(pil)
        tk_img = pil_to_tk(pil)
        if target == "content":
            self._content_img_tk = tk_img
            self.content_label.configure(image=tk_img)
        elif target == "style":
            self._style_img_tk = tk_img
            self.style_label.configure(image=tk_img)
        else:
            self._output_img_tk = tk_img
            self.output_label.configure(image=tk_img)

    def _run_transfer(self) -> None:
        if not self.content_path or not self.style_path.get() or not self.output_dir.get():
            messagebox.showerror("Missing input", "Please select content image, style image, and output folder.")
            return

        self.status_text.set("Processing...")
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

    def _worker(self) -> None:
        try:
            content = load_image_bgr(self.content_path)
            style = load_image_bgr(self.style_path.get())
            style = resize_to_match(style, content)

            out = transfer_style(
                content,
                style,
                color_strength=self.color_strength.get(),
                edge_strength=self.edge_strength.get(),
                detail_strength=self.detail_strength.get(),
            )

            base = os.path.splitext(os.path.basename(self.content_path))[0]
            out_name = f"{base}_stylized.png"
            out_path = os.path.join(self.output_dir.get(), out_name)
            save_image_bgr(out_path, out)
            self.output_path = out_path

            self.after(0, lambda: self._update_preview(out_path, target="output"))
            self.after(0, lambda: self.status_text.set("Completed"))
        except Exception as exc:
            self.after(0, lambda: self.status_text.set(f"Error: {exc}"))


if __name__ == "__main__":
    app = App()
    app.mainloop()
```

**Step 2: Manual run check**

Run: `python main.py`
Expected: UI opens, previews load, status updates from Idle -> Processing -> Completed

---

### Task 7: Verify end-to-end behavior

**Files:**
- No code changes

**Step 1: Install deps**

Run: `python -m pip install -r requirements.txt`
Expected: Install completes

**Step 2: Run tests**

Run: `python -m pytest -v`
Expected: PASS

**Step 3: Manual verification**

- Choose input folder, select content + style, select output folder, click Run
- Verify output file saved and output preview updated

---

## Notes
- User requested no git workflow; worktree step skipped.
- UI intentionally simple with directory selection + previews + status bar.

