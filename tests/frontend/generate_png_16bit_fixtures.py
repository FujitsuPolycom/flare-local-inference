# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record Pillow RGB8 output for 16-bit PNG inputs.

Pillow cannot write 16-bit RGB, RGBA, or grayscale-alpha PNGs, so the encoded
inputs are assembled here. The expected RGB bytes always come from Pillow.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from PIL import Image, __version__ as pillow_version


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode(width: int, height: int, color: int, rows: list[bytes], trns: bytes | None) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 16, color, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + (chunk(b"tRNS", trns) if trns is not None else b"")
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def samples(values: list[list[tuple[int, ...]]]) -> list[bytes]:
    return [b"".join(struct.pack(">" + "H" * len(pixel), *pixel) for pixel in row) for row in values]


cases = [
    # Grayscale without alpha clamps through Pillow's I;16 to RGB conversion.
    ("gray16", 0, [[(0x1234,), (0x00FE,)], [(0x0100,), (0x0000,)]], None),
    ("gray16-trns", 0, [[(0x1234,), (0x00FE,)], [(0x0100,), (0x0000,)]], struct.pack(">H", 0x1234)),
    ("gray-alpha16", 4,
        [[(0x1234, 0x80FF), (0x00FE, 0x0100)], [(0x0100, 0xFFFF), (0x0000, 0x0000)]], None),
    ("rgb16", 2,
        [[(0x1234, 0xABFF, 0x00FF), (0x00FE, 0x0100, 0xFFFF)],
         [(0x0100, 0xFFFF, 0x0000), (0x0000, 0x0000, 0x0000)]], None),
    ("rgba16", 6,
        [[(0x1234, 0xABFF, 0x00FF, 0x80FF), (0x00FE, 0x0100, 0xFFFF, 0x0100)],
         [(0x0100, 0x0000, 0xFFFF, 0xFFFF), (0x0000, 0x0000, 0x0000, 0x0000)]], None),
    ("rgb16-trns", 2,
        [[(0x1234, 0xABFF, 0x00FF), (0x1234, 0xABFF, 0x00FE)],
         [(0x00FE, 0x0100, 0xFFFF), (0x0100, 0xFFFF, 0x0000)]],
        struct.pack(">HHH", 0x1234, 0xABFF, 0x00FF)),
]


def main() -> None:
    if pillow_version != "12.3.0":
        raise RuntimeError(f"fixture generation requires Pillow 12.3.0, got {pillow_version}")
    root = Path(__file__).resolve().parent
    inputs = root / "png-16bit-inputs"
    inputs.mkdir(exist_ok=True)
    manifest = []
    for name, color, values, trns in cases:
        height = len(values)
        width = len(values[0])
        path = inputs / f"{name}.png"
        path.write_bytes(encode(width, height, color, samples(values), trns))
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgba = image.convert("RGBA")
        expected = path.with_suffix(".rgb")
        expected.write_bytes(rgb.tobytes())
        expected_rgba = path.with_suffix(".rgba")
        expected_rgba.write_bytes(rgba.tobytes())
        manifest.append({
            "name": name,
            "encoded": str(path.relative_to(root)),
            "rgb": str(expected.relative_to(root)),
            "rgba": str(expected_rgba.relative_to(root)),
            "width": width,
            "height": height,
        })
    (root / "png_16bit.json").write_text(
        json.dumps({"pillow": pillow_version, "cases": manifest}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
