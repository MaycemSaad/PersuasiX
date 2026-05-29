#!/usr/bin/env python3
"""Generate PNG icons for the PersuasiX Chrome extension."""

import struct
import zlib
from pathlib import Path


def create_png(width: int, height: int) -> bytes:
    """Create a simple PNG icon with a purple gradient 'P' logo."""

    def make_pixel(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        """Generate pixel color for position (x, y)."""
        cx, cy = w / 2, h / 2
        # Background: dark gradient
        bg_r = int(15 + (x / w) * 10)
        bg_g = int(23 + (y / h) * 10)
        bg_b = int(42 + (x / w) * 20)

        # Circle
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        radius = min(w, h) * 0.42

        if dist < radius:
            # Inside circle: purple gradient
            t = dist / radius
            r = int(99 + (1 - t) * 30)
            g = int(102 + (1 - t) * 20)
            b = int(241 + (1 - t) * 14)

            # Letter "P" region (simplified)
            rel_x = (x - cx) / radius
            rel_y = (y - cy) / radius

            in_letter = False
            # Vertical bar of P
            if -0.3 <= rel_x <= -0.05 and -0.55 <= rel_y <= 0.55:
                in_letter = True
            # Top horizontal bar of P
            if -0.3 <= rel_x <= 0.25 and -0.55 <= rel_y <= -0.3:
                in_letter = True
            # Right curve of P (simplified as rectangle)
            if 0.15 <= rel_x <= 0.35 and -0.55 <= rel_y <= 0.0:
                in_letter = True
            # Bottom horizontal of P bowl
            if -0.05 <= rel_x <= 0.25 and -0.1 <= rel_y <= 0.05:
                in_letter = True

            if in_letter:
                return (240, 240, 255, 255)

            return (min(r, 255), min(g, 255), min(b, 255), 255)
        else:
            return (bg_r, bg_g, bg_b, 0)

    # Generate raw pixel data
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"  # Filter byte
        for x in range(width):
            r, g, b, a = make_pixel(x, y, width, height)
            raw_data += struct.pack("BBBB", r, g, b, a)

    # PNG file structure
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png += make_chunk(b"IHDR", ihdr_data)
    # IDAT
    compressed = zlib.compress(raw_data)
    png += make_chunk(b"IDAT", compressed)
    # IEND
    png += make_chunk(b"IEND", b"")

    return png


if __name__ == "__main__":
    icons_dir = Path(__file__).parent
    for size in [16, 48, 128]:
        data = create_png(size, size)
        path = icons_dir / f"icon{size}.png"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Generated {path} ({len(data)} bytes)")
