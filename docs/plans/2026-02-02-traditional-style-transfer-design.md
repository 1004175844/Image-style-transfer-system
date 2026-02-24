# Traditional Style Transfer (No-Training) Design

**Goal:** Build a simple desktop app that performs traditional, no-training style transfer using two images: content + style.

**Approach Summary:** Use Reinhard color transfer in Lab space, then edge-preserving smoothing (bilateral), then a detail-boost blend from the content image. Provide a minimal Tkinter UI to select folders/images and show previews with a status bar.

**Tech Stack:** Python 3, Tkinter, Pillow (PIL), OpenCV (cv2), NumPy.

## Architecture
- `main.py`: App entry, window layout, event wiring, background worker.
- `core/transfer.py`: Style transfer pipeline (color transfer, smoothing, detail boost).
- `core/io.py`: Image load/save, resize, conversion helpers.
- `ui/widgets.py`: Small helpers for preview rendering.

## UI Layout
- Row 1: Input folder selector (Browse). This populates the content image list.
- Row 2: Content image dropdown/list. Selecting loads preview.
- Row 3: Style image selector (Browse). Single file path.
- Row 4: Output folder selector (Browse).
- Middle: Three preview boxes in a row: Content | Style | Output.
- Controls: 3 sliders (color strength, edge strength, detail boost) and a Run button.
- Bottom: Status bar (Idle / Processing / Done / Error).

## Data Flow
1. User selects input folder and content image.
2. User selects style image and output folder.
3. User tunes sliders and clicks Run.
4. Background worker runs `transfer(content, style, params)`.
5. Output saved to output folder and preview updated.
6. Status bar updated at each stage.

## Algorithm Detail
1. Load content (BGR/RGB) and style.
2. Resize style to content dimensions (preserve aspect, center crop if needed).
3. Convert both to Lab.
4. Reinhard transfer per channel:
   - Normalize channel: `(x - mean_c) * (std_s / std_c) + mean_s`
   - Clamp to valid Lab range.
   - Blend with original via `color_strength`.
5. Edge-preserving smoothing:
   - Bilateral filter with parameters driven by `edge_strength`.
6. Detail boost:
   - Extract detail layer from content: `detail = content - GaussianBlur(content)`.
   - Output = smoothed + detail_boost * detail.
7. Convert to display/save format.

## Error Handling
- Validate image file existence and formats.
- Ensure output folder exists or create it.
- Catch exceptions in worker thread and report in status bar.

## Testing (Lightweight)
- Manual checks: load images, run transfer, verify output saved.
- Parameter sanity: extreme slider values should not crash and output should remain valid.

## Files to Create
- `main.py`
- `core/transfer.py`
- `core/io.py`
- `ui/widgets.py`
- `requirements.txt`

