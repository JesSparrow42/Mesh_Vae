#!/usr/bin/env python3
"""Render a paper figure from point clouds exported by a trained MeshVAE.

This script never reads the poster image. Its input is the NPZ bundle produced
by ``export_model_reconstructions.py`` from a real checkpoint and held-out data.
It uses Pillow rather than Matplotlib to keep the Gefion dependency footprint
small and predictable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "input": (91, 108, 122, 230),
    "reconstruction": (8, 127, 140, 230),
    "ink": "#17324D",
    "muted": "#617581",
    "grid": "#DCE5E9",
    "panel": "#FBFCFD",
    "white": "#FFFFFF",
}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="Held-out full-body scan reconstructions")
    parser.add_argument("--vertical-axis", choices=("auto", "x", "y", "z"), default="auto")
    parser.add_argument("--front-axis", choices=("auto", "x", "y", "z"), default="auto")
    parser.add_argument("--max-points", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-participant-ids", action="store_true")
    return parser


def finite_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected point array with shape (N, 3), got {array.shape}.")
    array = array[np.isfinite(array).all(axis=1)]
    if len(array) < 10:
        raise ValueError("Point cloud contains fewer than 10 finite points.")
    return array


def robust_span(points: np.ndarray) -> np.ndarray:
    low, high = np.percentile(points, [1.0, 99.0], axis=0)
    return high - low


def infer_projection_axes(
    inputs: list[np.ndarray], vertical_axis: str, front_axis: str
) -> tuple[int, int, int]:
    combined = np.concatenate(inputs, axis=0)
    spans = robust_span(combined)
    vertical = int(np.argmax(spans)) if vertical_axis == "auto" else AXIS_INDEX[vertical_axis]
    remaining = [axis for axis in range(3) if axis != vertical]
    if front_axis == "auto":
        front = max(remaining, key=lambda axis: spans[axis])
    else:
        front = AXIS_INDEX[front_axis]
        if front == vertical:
            raise ValueError("Front and vertical axes must be different.")
    depth = next(axis for axis in range(3) if axis not in (vertical, front))
    return vertical, front, depth


def downsample(points: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = rng.choice(len(points), size=maximum, replace=False)
    return points[np.sort(indices)]


def centered_pair(original: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(original, axis=0)
    return original - center, reconstruction - center


def symmetric_limit(arrays: list[np.ndarray], axis: int, padding: float = 0.07) -> tuple[float, float]:
    values = np.concatenate([array[:, axis] for array in arrays])
    low, high = np.percentile(values, [0.5, 99.5])
    extent = max(abs(float(low)), abs(float(high)), 1e-6) * (1.0 + padding)
    return -extent, extent


def vertical_limit(arrays: list[np.ndarray], axis: int, padding: float = 0.04) -> tuple[float, float]:
    values = np.concatenate([array[:, axis] for array in arrays])
    low, high = np.percentile(values, [0.5, 99.5])
    margin = max(float(high - low), 1e-6) * padding
    return float(low - margin), float(high + margin)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    text_font: ImageFont.ImageFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=text_font, fill=fill)


def render_projection(
    points: np.ndarray,
    horizontal: int,
    vertical: int,
    color: tuple[int, int, int, int],
    horizontal_limit: tuple[float, float],
    vertical_limits: tuple[float, float],
    width: int,
    height: int,
) -> Image.Image:
    panel = Image.new("RGBA", (width, height), COLORS["panel"])
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    inset = 28
    x_low, x_high = horizontal_limit
    y_low, y_high = vertical_limits
    x_span = max(x_high - x_low, 1e-12)
    y_span = max(y_high - y_low, 1e-12)
    scale = min((width - 2 * inset) / x_span, (height - 2 * inset) / y_span)
    x_offset = (width - x_span * scale) / 2
    y_offset = (height - y_span * scale) / 2
    x_values = x_offset + (points[:, horizontal] - x_low) * scale
    y_values = height - (y_offset + (points[:, vertical] - y_low) * scale)
    coordinates = [
        (int(round(x_value)), int(round(y_value)))
        for x_value, y_value in zip(x_values, y_values)
        if 0 <= x_value < width and 0 <= y_value < height
    ]
    for x_value, y_value in coordinates:
        draw.ellipse(
            (x_value - 1, y_value - 1, x_value + 1, y_value + 1),
            fill=color,
        )
    panel.alpha_composite(overlay)
    border = ImageDraw.Draw(panel)
    border.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=10,
        outline=COLORS["grid"],
        width=3,
    )
    return panel


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = args.bundle.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else bundle_path.with_name("reconstruction_manifest.csv")
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Reconstruction manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"example_key", "target", "target_label", "chamfer", "selection_method"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Manifest is missing columns: {missing}")
    manifest = manifest.sort_values(["target", "example_key"]).reset_index(drop=True)

    archive = np.load(bundle_path)
    examples: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        key = str(row.example_key)
        original = finite_points(archive[f"{key}_input"])
        reconstruction = finite_points(archive[f"{key}_reconstruction"])
        original, reconstruction = centered_pair(original, reconstruction)
        examples.append(
            {
                "row": row,
                "input": original,
                "reconstruction": reconstruction,
            }
        )
    if not examples:
        raise ValueError("The reconstruction bundle contains no examples.")

    vertical, front, depth = infer_projection_axes(
        [example["input"] for example in examples], args.vertical_axis, args.front_axis
    )
    all_arrays = [
        array
        for example in examples
        for array in (example["input"], example["reconstruction"])
    ]
    front_limit = symmetric_limit(all_arrays, front)
    side_limit = symmetric_limit(all_arrays, depth)
    height_limit = vertical_limit(all_arrays, vertical)

    canvas_width = 2800
    left_margin, right_margin = 390, 80
    title_height, column_height, footer_height = 150, 85, 125
    column_gutter, row_gutter = 44, 42
    panel_width = (canvas_width - left_margin - right_margin - 2 * column_gutter) // 3
    panel_height = 650
    canvas_height = (
        title_height
        + column_height
        + len(examples) * panel_height
        + max(0, len(examples) - 1) * row_gutter
        + footer_height
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), COLORS["white"])
    draw = ImageDraw.Draw(canvas)
    title_font = font(56, bold=True)
    column_font = font(34, bold=True)
    row_font = font(30, bold=True)
    detail_font = font(24, bold=False)
    footer_font = font(23, bold=False)

    draw_centered_text(
        draw, canvas_width / 2, 40, args.title, title_font, COLORS["ink"]
    )
    column_titles = ("Held-out scan", "Reconstruction", "Reconstruction, side view")
    for column, title in enumerate(column_titles):
        x = left_margin + column * (panel_width + column_gutter)
        draw_centered_text(
            draw, x + panel_width / 2, title_height + 20, title, column_font, COLORS["ink"]
        )

    rng = np.random.default_rng(args.seed)
    panel_top = title_height + column_height
    for row_index, example in enumerate(examples):
        original = downsample(example["input"], args.max_points, rng)
        reconstruction = downsample(example["reconstruction"], args.max_points, rng)
        projections = (
            render_projection(
                original,
                front,
                vertical,
                COLORS["input"],
                front_limit,
                height_limit,
                panel_width,
                panel_height,
            ),
            render_projection(
                reconstruction,
                front,
                vertical,
                COLORS["reconstruction"],
                front_limit,
                height_limit,
                panel_width,
                panel_height,
            ),
            render_projection(
                reconstruction,
                depth,
                vertical,
                COLORS["reconstruction"],
                side_limit,
                height_limit,
                panel_width,
                panel_height,
            ),
        )
        y = panel_top + row_index * (panel_height + row_gutter)
        for column, projection in enumerate(projections):
            x = left_margin + column * (panel_width + column_gutter)
            canvas.paste(projection.convert("RGB"), (x, y))

        row = example["row"]
        label = str(row.target_label).replace("-", " ").capitalize()
        details = "Median-error held-out example"
        if args.show_participant_ids and hasattr(row, "participant_id"):
            details += f"\nID: {row.participant_id}"
        label_y = y + panel_height / 2 - 42
        draw.multiline_text(
            (55, label_y),
            label,
            font=row_font,
            fill=COLORS["ink"],
            spacing=8,
        )
        draw.multiline_text(
            (55, label_y + 52),
            details,
            font=detail_font,
            fill=COLORS["muted"],
            spacing=7,
        )

    method = str(manifest["selection_method"].iloc[0]).replace("_", " ")
    footer = (
        f"Held-out examples selected by {method}. Deterministic posterior-mean reconstructions.\n"
        "Plaque status was not supplied to the scan-only encoder; these examples do not test class separation."
    )
    footer_y = canvas_height - footer_height + 24
    footer_box = draw.multiline_textbbox((0, 0), footer, font=footer_font, spacing=7, align="center")
    footer_width = footer_box[2] - footer_box[0]
    draw.multiline_text(
        ((canvas_width - footer_width) / 2, footer_y),
        footer,
        font=footer_font,
        fill=COLORS["muted"],
        spacing=7,
        align="center",
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "reconstruction_examples.png"
    tiff = output_dir / "reconstruction_examples.tiff"
    pdf = output_dir / "reconstruction_examples.pdf"
    canvas.save(png, dpi=(args.dpi, args.dpi), optimize=True)
    canvas.save(tiff, dpi=(args.dpi, args.dpi), compression="tiff_lzw")
    canvas.save(pdf, "PDF", resolution=float(args.dpi))

    export_metadata_path = bundle_path.with_name("reconstruction_export.json")
    export_metadata = {}
    if export_metadata_path.exists():
        export_metadata = json.loads(export_metadata_path.read_text(encoding="utf-8"))
    gender = export_metadata.get("gender", "sex-stratified")
    caption = (
        f"Representative held-out {gender} full-body point-cloud inputs and deterministic MeshVAE "
        "reconstructions. Within each plaque target group, examples were selected by reconstruction "
        "error rather than visual appearance. Front and side projections share fixed axis scales. "
        "The examples illustrate reconstruction fidelity and are not evidence of plaque separation."
    )
    (output_dir / "reconstruction_examples_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    print(f"[AXES] vertical={vertical} front={front} depth={depth}")
    print(f"[OUT] {png}")
    print(f"[OUT] {tiff}")
    print(f"[OUT] {pdf}")


if __name__ == "__main__":
    main()
