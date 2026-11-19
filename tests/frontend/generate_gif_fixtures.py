# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze small GIF inputs with Pillow's RGB and RGBA conversions of their first frame."""
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import GifImagePlugin, Image


def palette_image(pixels, colors):
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8), "P")
    image.putpalette([channel for color in colors for channel in color])
    return image


def index_grid(width, height, count, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, count, (height, width), dtype=np.uint8)


def colors(count, seed):
    rng = np.random.default_rng(seed)
    return [tuple(int(v) for v in rng.integers(0, 256, 3)) for _ in range(count)]


def save(image, **options):
    buffer = io.BytesIO()
    image.save(buffer, format="GIF", **options)
    return buffer.getvalue()


def frame_at(canvas, frame, offset, **params):
    """A single-frame GIF whose image is `frame` placed at `offset` on a `canvas`-sized screen.

    Pillow's public writer always places a frame at the origin of a screen of its own
    size, so this uses its frame writers directly; the Pillow version check guards them.
    """
    buffer = io.BytesIO()
    info = {k: v for k, v in params.items() if k == "transparency"}
    for part in GifImagePlugin._get_global_header(canvas, info):
        buffer.write(part)
    GifImagePlugin._write_frame_data(buffer, frame, offset, params)
    buffer.write(b";")
    return buffer.getvalue()


def without_table(frame):
    """A GIF with neither a global nor a local color table; Pillow reads it as mode L."""
    buffer = io.BytesIO()
    buffer.write(b"GIF89a" + struct.pack("<HHBBB", *frame.size, 0, 0, 0))
    GifImagePlugin._write_frame_data(buffer, frame, (0, 0), {})
    buffer.write(b";")
    return buffer.getvalue()


def cases():
    """Yields (name, encoded bytes, animated)."""
    palette = colors(16, 1)
    other = colors(16, 2)
    identity = [(level, level, level) for level in range(256)]
    still = save(palette_image(index_grid(11, 7, 16, 10), palette), interlace=False)
    yield "global-palette", still, False
    yield "local-palette", frame_at(
        palette_image(index_grid(9, 6, 16, 11), palette),
        palette_image(index_grid(9, 6, 16, 12), other), (0, 0), include_color_table=True), False
    yield "transparency-index", save(
        palette_image(index_grid(13, 9, 16, 13), palette), interlace=False, transparency=5), False
    screen = palette_image(np.zeros((12, 20), np.uint8), palette)
    small = palette_image(index_grid(7, 5, 16, 14), palette)
    yield "frame-offset", frame_at(screen, small, (6, 4)), False
    yield "frame-offset-transparency", frame_at(screen, small, (6, 4), transparency=3), False
    yield "frame-past-screen", frame_at(
        palette_image(np.zeros((6, 8), np.uint8), palette), small, (5, 3)), False
    yield "empty-screen", still[:6] + bytes(4) + still[10:], False
    yield "interlaced", save(palette_image(index_grid(21, 19, 16, 15), palette), interlace=True), False
    yield "interlaced-flat", save(Image.new("P", (31, 19), 0), interlace=True), False
    yield "gray-identity-table", save(
        Image.fromarray(index_grid(17, 18, 256, 16), "L"), interlace=True, optimize=False), False
    yield "gray-no-table", without_table(Image.fromarray(index_grid(9, 5, 256, 19), "L")), False
    yield "gray-index-past-table", frame_at(
        palette_image(np.zeros((5, 9), np.uint8), identity[:4]),
        Image.fromarray(index_grid(9, 5, 256, 20), "L"), (0, 0)), False
    yield "palette-index-past-table", frame_at(
        palette_image(np.zeros((5, 9), np.uint8), palette[:4]),
        palette_image(index_grid(9, 5, 16, 23), palette), (0, 0)), False
    yield "local-identity-over-global", frame_at(
        palette_image(index_grid(9, 6, 16, 21), palette),
        palette_image(index_grid(9, 6, 16, 22), identity), (0, 0), include_color_table=True), False
    yield "missing-trailer", still[:-1], False
    first = palette_image(index_grid(8, 6, 16, 17), palette)
    second = palette_image(index_grid(8, 6, 16, 18), palette)
    yield "two-frames", save(first, save_all=True, append_images=[second], duration=100, loop=0), True


def main():
    if Image.__version__ != "12.3.0":
        raise RuntimeError(f"fixture generation requires Pillow 12.3.0, got {Image.__version__}")
    root = Path(__file__).resolve().parent
    inputs = root / "gif-inputs"
    inputs.mkdir(exist_ok=True)
    manifest = []
    for name, data, animated in cases():
        path = inputs / f"{name}.gif"
        path.write_bytes(data)
        case = {"name": name, "encoded": str(path.relative_to(root)), "animated": animated}
        if not animated:
            with Image.open(path) as image:
                mode = image.mode
                rgb = image.convert("RGB")
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
            expected = path.with_suffix(".rgb")
            expected.write_bytes(rgb.tobytes())
            expected_rgba = path.with_suffix(".rgba")
            expected_rgba.write_bytes(rgba.tobytes())
            case.update(mode=mode, rgb=str(expected.relative_to(root)), rgba=str(expected_rgba.relative_to(root)),
                width=rgb.width, height=rgb.height)
        manifest.append(case)
    (root / "gif.json").write_text(json.dumps({"pillow": Image.__version__, "cases": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
