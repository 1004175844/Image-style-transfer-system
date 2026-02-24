from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image
from tkinter import filedialog, messagebox, ttk

from core.io import is_image_file, load_image_bgr, resize_to_match, save_image_bgr
from core.metrics import compute_metrics
from core.transfer import transfer_style
from ui.metrics_panel import MetricsPanel
from ui.widgets import fit_image, pil_to_tk


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("图像风格迁移系统")
        self.geometry("1320x920")
        self.minsize(1160, 800)

        self.input_dir = tk.StringVar()
        self.style_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status_text = tk.StringVar(value="就绪")

        self.style_strength = tk.DoubleVar(value=1.0)
        self.structure_strength = tk.DoubleVar(value=0.5)
        self.detail_fidelity = tk.DoubleVar(value=0.4)
        self.style_value_text = tk.StringVar()
        self.structure_value_text = tk.StringVar()
        self.detail_value_text = tk.StringVar()

        self.content_files: list[str] = []
        self.content_path: Optional[str] = None
        self.output_path: Optional[str] = None

        self._content_img_tk = None
        self._style_img_tk = None
        self._output_img_tk = None

        self._interactive_widgets: list[tk.Widget] = []
        self._widget_state_cache: dict[tk.Widget, str] = {}

        self.metrics_panel: Optional[MetricsPanel] = None

        self._palette = {
            "bg": "#f3f5f8",
            "surface": "#ffffff",
            "muted_surface": "#f7f9fc",
            "border": "#d8dfe7",
            "text": "#1f2a37",
            "subtext": "#5b6878",
            "primary": "#1f6feb",
            "primary_hover": "#1457be",
            "success": "#0f8b5f",
        }

        self._bind_slider_values()
        self._build_ui()

    def _build_ui(self) -> None:
        self._configure_theme()

        root = ttk.Frame(self, style="App.TFrame", padding=(16, 14, 16, 12))
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root, style="Header.TFrame", padding=(16, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text="图像风格迁移系统", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="选择素材并执行处理，结果和评价图会自动同步更新。",
            style="HeaderSubTitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        tabs_wrap = ttk.Frame(root, style="App.TFrame")
        tabs_wrap.pack(fill=tk.BOTH, expand=True, pady=(10, 6))

        notebook = ttk.Notebook(tabs_wrap, style="App.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_main = ttk.Frame(notebook, style="App.TFrame", padding=(6, 8, 6, 6))
        tab_metrics = ttk.Frame(notebook, style="App.TFrame", padding=(6, 8, 6, 6))
        notebook.add(tab_main, text="处理")
        notebook.add(tab_metrics, text="评价指标")

        self._build_workbench_tab(tab_main)
        self._build_metrics_tab(tab_metrics)

        status_wrap = ttk.Frame(root, style="StatusWrap.TFrame")
        status_wrap.pack(fill=tk.X)
        ttk.Label(status_wrap, textvariable=self.status_text, style="Status.TLabel", anchor="w").pack(fill=tk.X, padx=1, pady=1)

    def _build_workbench_tab(self, parent: ttk.Frame) -> None:
        body = ttk.Frame(parent, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1, minsize=420)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left_col = ttk.Frame(body, style="App.TFrame")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_col = ttk.Frame(body, style="App.TFrame")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._build_path_card(left_col)
        self._build_parameter_card(left_col)
        self._build_action_card(left_col)
        self._build_preview_card(right_col)

    def _build_metrics_tab(self, parent: ttk.Frame) -> None:
        self.metrics_panel = MetricsPanel(parent, palette=self._palette)
        self.metrics_panel.pack(fill=tk.BOTH, expand=True)

    def _configure_theme(self) -> None:
        self.configure(bg=self._palette["bg"])

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Microsoft YaHei UI", size=10)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self._palette["bg"])
        style.configure("Header.TFrame", background=self._palette["surface"], relief="flat")
        style.configure("CardBorder.TFrame", background=self._palette["border"])
        style.configure("Card.TFrame", background=self._palette["surface"])
        style.configure("PreviewBorder.TFrame", background=self._palette["border"])
        style.configure("PreviewCard.TFrame", background=self._palette["muted_surface"])
        style.configure("ImageBox.TFrame", background=self._palette["surface"])
        style.configure("StatusWrap.TFrame", background=self._palette["border"])

        style.configure("App.TNotebook", background=self._palette["bg"], borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("App.TNotebook.Tab", padding=(14, 7), font=("Microsoft YaHei UI", 10))
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", self._palette["surface"]), ("!selected", "#e9eef6")],
            foreground=[("selected", self._palette["primary"]), ("!selected", self._palette["subtext"])],
            padding=[("selected", (20, 11)), ("!selected", (14, 7))],
        )

        style.configure(
            "HeaderTitle.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["text"],
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "HeaderSubTitle.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["subtext"],
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["text"],
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "CardHint.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["subtext"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Field.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["text"],
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Value.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["primary"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "PreviewTitle.TLabel",
            background=self._palette["muted_surface"],
            foreground=self._palette["text"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Placeholder.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["subtext"],
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background=self._palette["surface"],
            foreground=self._palette["subtext"],
            font=("Microsoft YaHei UI", 10),
            padding=(12, 8),
        )

        style.configure(
            "Primary.TButton",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="#ffffff",
            background=self._palette["primary"],
            padding=(12, 8),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", self._palette["primary_hover"]), ("disabled", "#9fb6de")],
            foreground=[("disabled", "#e8eef8")],
        )

        style.configure(
            "Secondary.TButton",
            font=("Microsoft YaHei UI", 9),
            foreground=self._palette["text"],
            background="#eef2f7",
            padding=(8, 7),
            borderwidth=0,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#dde5f0"), ("disabled", "#eef2f7")],
            foreground=[("disabled", "#8d99a7")],
        )

        style.configure(
            "Business.TEntry",
            fieldbackground=self._palette["surface"],
            foreground=self._palette["text"],
            padding=(8, 7),
        )
        style.configure(
            "Business.TCombobox",
            fieldbackground=self._palette["surface"],
            foreground=self._palette["text"],
            padding=(6, 5),
        )
        style.configure("Modern.Horizontal.TScale", background=self._palette["surface"], troughcolor="#dce4ef")
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor="#dce4ef",
            background=self._palette["primary"],
            borderwidth=0,
        )

    def _create_card(self, parent: ttk.Frame, title: str, hint: str) -> ttk.Frame:
        border = ttk.Frame(parent, style="CardBorder.TFrame")
        border.pack(fill=tk.X, pady=(0, 12))
        card = ttk.Frame(border, style="Card.TFrame", padding=(16, 14))
        card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=hint, style="CardHint.TLabel").pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill=tk.X, pady=(12, 0))
        return body

    def _build_path_card(self, parent: ttk.Frame) -> None:
        body = self._create_card(parent, "素材与路径", "选择内容图、风格图和输出目录。")
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="输入目录", style="Field.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        input_entry = ttk.Entry(body, textvariable=self.input_dir, style="Business.TEntry")
        input_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=(0, 8))
        input_btn = ttk.Button(body, text="浏览", style="Secondary.TButton", command=self._pick_input_dir)
        input_btn.grid(row=0, column=2, pady=(0, 8))

        ttk.Label(body, text="内容图像", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 8))
        self.content_combo = ttk.Combobox(body, state="readonly", style="Business.TCombobox")
        self.content_combo.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(0, 8))
        self.content_combo.bind("<<ComboboxSelected>>", self._on_content_selected)

        ttk.Label(body, text="风格图像", style="Field.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 8))
        style_entry = ttk.Entry(body, textvariable=self.style_path, style="Business.TEntry")
        style_entry.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(0, 8))
        style_btn = ttk.Button(body, text="浏览", style="Secondary.TButton", command=self._pick_style_file)
        style_btn.grid(row=2, column=2, pady=(0, 8))

        ttk.Label(body, text="输出目录", style="Field.TLabel").grid(row=3, column=0, sticky="w")
        output_entry = ttk.Entry(body, textvariable=self.output_dir, style="Business.TEntry")
        output_entry.grid(row=3, column=1, sticky="ew", padx=(8, 8))
        output_btn = ttk.Button(body, text="浏览", style="Secondary.TButton", command=self._pick_output_dir)
        output_btn.grid(row=3, column=2)

        self._interactive_widgets.extend(
            [input_entry, input_btn, self.content_combo, style_entry, style_btn, output_entry, output_btn]
        )

    def _build_parameter_card(self, parent: ttk.Frame) -> None:
        body = self._create_card(parent, "参数设置", "调整风格迁移强度与细节表现。")
        self._add_slider_row(body, "风格强度", self.style_strength, self.style_value_text)
        self._add_slider_row(body, "结构强度", self.structure_strength, self.structure_value_text)
        self._add_slider_row(body, "细节保真", self.detail_fidelity, self.detail_value_text)

    def _add_slider_row(
        self, parent: ttk.Frame, title: str, variable: tk.DoubleVar, value_text: tk.StringVar
    ) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(0, 10))

        head = ttk.Frame(row, style="Card.TFrame")
        head.pack(fill=tk.X)
        ttk.Label(head, text=title, style="Field.TLabel").pack(side=tk.LEFT)
        ttk.Label(head, textvariable=value_text, style="Value.TLabel").pack(side=tk.RIGHT)

        scale = ttk.Scale(row, from_=0, to=1, orient=tk.HORIZONTAL, variable=variable, style="Modern.Horizontal.TScale")
        scale.pack(fill=tk.X, pady=(6, 0))
        self._interactive_widgets.append(scale)

    def _build_action_card(self, parent: ttk.Frame) -> None:
        body = self._create_card(parent, "执行处理", "按当前设置生成结果并自动更新评价图。")

        self.run_btn = ttk.Button(body, text="开始处理", style="Primary.TButton", command=self._run_transfer)
        self.run_btn.pack(fill=tk.X)
        self.progress = ttk.Progressbar(body, mode="indeterminate", style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=(10, 0))

        self._interactive_widgets.append(self.run_btn)

    def _build_preview_card(self, parent: ttk.Frame) -> None:
        border = ttk.Frame(parent, style="CardBorder.TFrame")
        border.pack(fill=tk.BOTH, expand=True)
        card = ttk.Frame(border, style="Card.TFrame", padding=(16, 14))
        card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        ttk.Label(card, text="结果预览", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            card,
            text="并排查看内容图、风格图和输出图。",
            style="CardHint.TLabel",
        ).pack(anchor="w", pady=(2, 10))

        previews = ttk.Frame(card, style="Card.TFrame")
        previews.pack(fill=tk.BOTH, expand=True)
        for idx in range(3):
            previews.columnconfigure(idx, weight=1)
        previews.rowconfigure(0, weight=1)

        self.content_label = self._create_preview_slot(previews, 0, "内容图")
        self.style_label = self._create_preview_slot(previews, 1, "风格图")
        self.output_label = self._create_preview_slot(previews, 2, "输出图")

    def _create_preview_slot(self, parent: ttk.Frame, column: int, title: str) -> ttk.Label:
        border = ttk.Frame(parent, style="PreviewBorder.TFrame")
        border.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0 if column == 2 else 6))
        slot = ttk.Frame(border, style="PreviewCard.TFrame", padding=(10, 10))
        slot.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        ttk.Label(slot, text=title, style="PreviewTitle.TLabel").pack(anchor="w", pady=(0, 8))

        img_box = ttk.Frame(slot, style="ImageBox.TFrame", width=280, height=280)
        img_box.pack(fill=tk.BOTH, expand=True)
        img_box.pack_propagate(False)

        label = ttk.Label(img_box, text="暂无图像", style="Placeholder.TLabel", anchor="center")
        label.pack(fill=tk.BOTH, expand=True)
        return label

    def _bind_slider_values(self) -> None:
        bindings = (
            (self.style_strength, self.style_value_text),
            (self.structure_strength, self.structure_value_text),
            (self.detail_fidelity, self.detail_value_text),
        )
        for src, dst in bindings:
            src.trace_add("write", lambda *_args, s=src, d=dst: d.set(f"{s.get():.2f}"))
            dst.set(f"{src.get():.2f}")

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
            self.status_text.set(f"已加载 {len(files)} 张图像")
        else:
            self.content_path = None
            self.content_label.configure(image="", text="暂无图像")
            self.status_text.set("输入目录中未发现图像文件")

    def _on_content_selected(self, _evt=None) -> None:
        idx = self.content_combo.current()
        if idx < 0:
            return
        self.content_path = self.content_files[idx]
        self._update_preview(self.content_path, target="content")

    def _pick_style_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp")])
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
        pil = fit_image(pil, size=(280, 280))
        tk_img = pil_to_tk(pil)
        if target == "content":
            self._content_img_tk = tk_img
            self.content_label.configure(image=tk_img, text="")
        elif target == "style":
            self._style_img_tk = tk_img
            self.style_label.configure(image=tk_img, text="")
        else:
            self._output_img_tk = tk_img
            self.output_label.configure(image=tk_img, text="")

    def _run_transfer(self) -> None:
        if not self.content_path or not self.style_path.get() or not self.output_dir.get():
            messagebox.showerror("输入不完整", "请先选择内容图、风格图和输出目录。")
            return

        self.status_text.set("处理中，请稍候...")
        self._set_running(True)
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
                color_strength=self.style_strength.get(),
                edge_strength=self.structure_strength.get(),
                detail_strength=self.detail_fidelity.get(),
            )

            base = os.path.splitext(os.path.basename(self.content_path))[0]
            out_name = f"{base}_result.png"
            out_path = os.path.join(self.output_dir.get(), out_name)
            save_image_bgr(out_path, out)
            self.output_path = out_path

            metric_result = compute_metrics(content, style, out)

            self.after(0, lambda: self._update_preview(out_path, target="output"))
            self.after(0, lambda result=metric_result: self._on_metrics_ready(result))
            self.after(0, lambda: self.status_text.set(f"处理完成：{out_name}"))
        except Exception as exc:
            err_msg = str(exc)
            self.after(0, lambda msg=err_msg: self.status_text.set(f"处理失败：{msg}"))
            self.after(0, lambda msg=err_msg: messagebox.showerror("处理失败", msg))
        finally:
            self.after(0, lambda: self._set_running(False))

    def _on_metrics_ready(self, metric_result: dict[str, object]) -> None:
        if not self.metrics_panel:
            return
        self.metrics_panel.update_metrics(metric_result)

    def _set_running(self, running: bool) -> None:
        for widget in self._interactive_widgets:
            try:
                if running:
                    self._widget_state_cache[widget] = str(widget.cget("state"))
                    widget.configure(state=tk.DISABLED)
                else:
                    previous_state = self._widget_state_cache.get(widget, tk.NORMAL)
                    widget.configure(state=previous_state)
            except tk.TclError:
                pass
        if not running:
            self._widget_state_cache.clear()
        if running:
            self.progress.start(14)
        else:
            self.progress.stop()


if __name__ == "__main__":
    app = App()
    app.mainloop()
