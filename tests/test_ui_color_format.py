from pathlib import Path
import re


def test_tkinter_canvas_hex_colors_use_6_digits():
    panel_path = Path("ui/metrics_panel.py")
    text = panel_path.read_text(encoding="utf-8")
    invalid_colors = re.findall(r"#[0-9a-fA-F]{8}", text)
    assert invalid_colors == []
