from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk


RADAR_ORDER = [
    "content_consistency",
    "style_alignment",
    "texture_coherence",
    "structure_stability",
    "detail_fidelity",
    "color_balance",
]

RADAR_LABELS = {
    "content_consistency": "内容一致性",
    "style_alignment": "风格匹配度",
    "texture_coherence": "纹理协调度",
    "structure_stability": "结构稳定性",
    "detail_fidelity": "细节保真度",
    "color_balance": "色彩平衡度",
}

COMPARE_ORDER = [
    "brightness_mean",
    "contrast_std",
    "saturation_mean",
    "sharpness",
    "entropy",
]

COMPARE_LABELS = {
    "brightness_mean": "亮度",
    "contrast_std": "对比度",
    "saturation_mean": "饱和度",
    "sharpness": "清晰度",
    "entropy": "信息熵",
}

STYLE_MATCH_ORDER = [
    "color_distribution_match",
    "saturation_match",
    "texture_match",
    "entropy_match",
]

STYLE_MATCH_LABELS = {
    "color_distribution_match": "色彩分布匹配",
    "saturation_match": "饱和度匹配",
    "texture_match": "纹理匹配",
    "entropy_match": "信息熵匹配",
}


def _blend_hex(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(round(r1 + (r2 - r1) * t))
    g = int(round(g1 + (g2 - g1) * t))
    b = int(round(b1 + (b2 - b1) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


class MetricsPanel(ttk.Frame):
    def __init__(self, parent: ttk.Frame, palette: dict[str, str]) -> None:
        super().__init__(parent, style="App.TFrame")
        self._palette = palette
        self._radar_scores = {name: 0.0 for name in RADAR_ORDER}
        self._triple_compare = {name: {"content": 0.0, "style": 0.0, "output": 0.0} for name in COMPARE_ORDER}
        self._style_match = {name: 0.0 for name in STYLE_MATCH_ORDER}

        top = ttk.Frame(self, style="App.TFrame")
        top.pack(fill=tk.BOTH, expand=True)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)

        bottom = ttk.Frame(self, style="App.TFrame")
        bottom.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(0, weight=1)

        self._radar_canvas = self._create_chart(top, 0, "综合评价雷达图")
        self._bar_canvas = self._create_chart(top, 1, "内容 / 风格 / 输出 对比图")
        self._bubble_canvas = self._create_chart(bottom, 0, "风格匹配分解图")

        self._radar_canvas.bind("<Configure>", lambda _e: self._draw_all())
        self._bar_canvas.bind("<Configure>", lambda _e: self._draw_all())
        self._bubble_canvas.bind("<Configure>", lambda _e: self._draw_all())

        self._draw_all()

    def _create_chart(self, parent: ttk.Frame, col: int, title: str) -> tk.Canvas:
        border = ttk.Frame(parent, style="CardBorder.TFrame")
        border.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 4, 0 if col == 1 else 0))
        body = ttk.Frame(border, style="Card.TFrame", padding=(10, 8))
        body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        ttk.Label(body, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 6))
        canvas = tk.Canvas(body, bg=self._palette["surface"], highlightthickness=0, relief=tk.FLAT)
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas

    def update_metrics(self, metric_result: dict[str, object]) -> None:
        scores = metric_result.get("scores", {})
        triple_compare = metric_result.get("triple_compare", {})
        style_match = metric_result.get("style_match", {})

        if isinstance(scores, dict):
            for name in RADAR_ORDER:
                self._radar_scores[name] = float(scores.get(name, 0.0))

        if isinstance(triple_compare, dict):
            for name in COMPARE_ORDER:
                payload = triple_compare.get(name, {})
                if isinstance(payload, dict):
                    self._triple_compare[name] = {
                        "content": float(payload.get("content", 0.0)),
                        "style": float(payload.get("style", 0.0)),
                        "output": float(payload.get("output", 0.0)),
                    }

        if isinstance(style_match, dict):
            for name in STYLE_MATCH_ORDER:
                self._style_match[name] = float(style_match.get(name, 0.0))

        self._draw_all()

    def _draw_all(self) -> None:
        self._draw_radar()
        self._draw_compare_bars()
        self._draw_style_bubbles()

    def _draw_radar(self) -> None:
        c = self._radar_canvas
        c.delete("all")
        width = max(c.winfo_width(), 2)
        height = max(c.winfo_height(), 2)
        if width < 90 or height < 90:
            return

        cx = width * 0.5
        cy = height * 0.54
        radius = min(width, height) * 0.34
        n = len(RADAR_ORDER)

        for layer in range(1, 6):
            r = radius * (layer / 5.0)
            ring = []
            for i in range(n):
                angle = -math.pi / 2 + i * (2 * math.pi / n)
                ring.extend([cx + r * math.cos(angle), cy + r * math.sin(angle)])
            c.create_polygon(ring, outline="#d6deea", fill="")

        for i, key in enumerate(RADAR_ORDER):
            angle = -math.pi / 2 + i * (2 * math.pi / n)
            x1 = cx + radius * math.cos(angle)
            y1 = cy + radius * math.sin(angle)
            c.create_line(cx, cy, x1, y1, fill="#d6deea")
            lx = cx + (radius + 18) * math.cos(angle)
            ly = cy + (radius + 18) * math.sin(angle)
            c.create_text(lx, ly, text=RADAR_LABELS[key], fill=self._palette["subtext"], font=("Microsoft YaHei UI", 9))

        points = []
        for i, key in enumerate(RADAR_ORDER):
            ratio = max(0.0, min(self._radar_scores[key] / 100.0, 1.0))
            angle = -math.pi / 2 + i * (2 * math.pi / n)
            points.extend([cx + radius * ratio * math.cos(angle), cy + radius * ratio * math.sin(angle)])
        c.create_polygon(points, fill="#dbe8ff", outline=self._palette["primary"], width=2)

    def _draw_compare_bars(self) -> None:
        c = self._bar_canvas
        c.delete("all")
        width = max(c.winfo_width(), 2)
        height = max(c.winfo_height(), 2)
        if width < 200 or height < 160:
            return

        pad_l = 28
        pad_r = 14
        pad_t = 20
        pad_b = 52
        x0 = pad_l
        y0 = pad_t
        x1 = width - pad_r
        y1 = height - pad_b

        c.create_line(x0, y1, x1, y1, fill="#c8d3e2")
        c.create_line(x0, y0, x0, y1, fill="#c8d3e2")
        for tick in range(0, 101, 20):
            yy = y1 - (y1 - y0) * (tick / 100.0)
            c.create_line(x0, yy, x1, yy, fill="#edf2f8")
            c.create_text(x0 - 4, yy, text=str(tick), anchor="e", fill=self._palette["subtext"], font=("Microsoft YaHei UI", 8))

        c.create_rectangle(x0 + 8, 4, x0 + 20, 14, fill="#5b8ff9", outline="")
        c.create_text(x0 + 24, 9, text="内容图", anchor="w", fill=self._palette["subtext"], font=("Microsoft YaHei UI", 8))
        c.create_rectangle(x0 + 76, 4, x0 + 88, 14, fill="#19a974", outline="")
        c.create_text(x0 + 92, 9, text="风格图", anchor="w", fill=self._palette["subtext"], font=("Microsoft YaHei UI", 8))
        c.create_rectangle(x0 + 144, 4, x0 + 156, 14, fill="#f39c12", outline="")
        c.create_text(x0 + 160, 9, text="输出图", anchor="w", fill=self._palette["subtext"], font=("Microsoft YaHei UI", 8))

        n = len(COMPARE_ORDER)
        slot_w = (x1 - x0) / max(n, 1)
        bar_w = max(6.0, slot_w * 0.18)

        for i, key in enumerate(COMPARE_ORDER):
            triple = self._triple_compare[key]
            v_content = max(0.0, float(triple["content"]))
            v_style = max(0.0, float(triple["style"]))
            v_out = max(0.0, float(triple["output"]))
            scale = max(v_content, v_style, v_out, 1e-6)
            content_norm = v_content / scale
            style_norm = v_style / scale
            out_norm = v_out / scale

            center = x0 + slot_w * (i + 0.5)
            left_x0 = center - bar_w * 1.7
            left_x1 = center - bar_w * 0.7
            mid_x0 = center - bar_w * 0.5
            mid_x1 = center + bar_w * 0.5
            right_x0 = center + bar_w * 0.7
            right_x1 = center + bar_w * 1.7

            content_h = (y1 - y0) * content_norm
            style_h = (y1 - y0) * style_norm
            out_h = (y1 - y0) * out_norm

            c.create_rectangle(left_x0, y1 - content_h, left_x1, y1, fill="#5b8ff9", outline="")
            c.create_rectangle(mid_x0, y1 - style_h, mid_x1, y1, fill="#19a974", outline="")
            c.create_rectangle(right_x0, y1 - out_h, right_x1, y1, fill="#f39c12", outline="")

            c.create_text(center, y1 + 12, text=COMPARE_LABELS[key], fill=self._palette["subtext"], font=("Microsoft YaHei UI", 8))
            c.create_text(
                center,
                y1 + 25,
                text=f"{v_content:.1f}/{v_style:.1f}/{v_out:.1f}",
                fill=self._palette["subtext"],
                font=("Microsoft YaHei UI", 8),
            )

    def _draw_style_bubbles(self) -> None:
        c = self._bubble_canvas
        c.delete("all")
        width = max(c.winfo_width(), 2)
        height = max(c.winfo_height(), 2)
        if width < 200 or height < 140:
            return

        cols = 2
        rows = 2
        pad_l = 12
        pad_r = 12
        pad_t = 12
        pad_b = 12
        cell_w = (width - pad_l - pad_r) / cols
        cell_h = (height - pad_t - pad_b) / rows

        for idx, key in enumerate(STYLE_MATCH_ORDER):
            row = idx // cols
            col = idx % cols
            x0 = pad_l + col * cell_w
            y0 = pad_t + row * cell_h
            x1 = x0 + cell_w - 8
            y1 = y0 + cell_h - 8

            c.create_rectangle(x0, y0, x1, y1, fill="#f5f8fd", outline="")
            value = max(0.0, min(self._style_match[key], 100.0))

            cx = (x0 + x1) * 0.5
            cy = y0 + (y1 - y0) * 0.58
            radius = max(8.0, min(cell_w, cell_h) * 0.14 + min(cell_w, cell_h) * 0.2 * (value / 100.0))

            fill = _blend_hex("#d8e5fb", self._palette["primary"], value / 100.0)
            c.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=fill, outline="")

            c.create_text(cx, y0 + 14, text=STYLE_MATCH_LABELS[key], fill=self._palette["subtext"], font=("Microsoft YaHei UI", 9, "bold"))
            c.create_text(cx, cy, text=f"{value:.1f}", fill="#ffffff", font=("Microsoft YaHei UI", 10, "bold"))
