"""Render 5 studio cyc backgrounds matching the reference style.

Reference: white/grey seamless cyc wall + concrete or tile floor + soft overhead
spotlights. These are the *target scenes* the harmonize stage matches the car to.

Run:
    .venv/bin/python -m scripts.render_backgrounds
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app" / "assets" / "backgrounds"
ASSETS.mkdir(parents=True, exist_ok=True)

# Production canvas: 1920x1280 (3:2)
W, H = 1920, 1280


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _smooth_noise(size: tuple[int, int], seed: int = 0, scale: int = 16) -> Image.Image:
    """Value noise. Generate a small random grid then upscale + blur."""
    w, h = size
    rng = np.random.default_rng(seed)
    sw, sh = max(w // scale, 4), max(h // scale, 4)
    small = rng.uniform(120, 200, size=(sh, sw)).astype(np.float32)
    img = Image.fromarray(small.astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def _vertical_gradient(
    size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> Image.Image:
    w, h = size
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    top_arr = np.array(top, dtype=np.float32)
    bot_arr = np.array(bottom, dtype=np.float32)
    col = (1 - t) * top_arr + t * bot_arr  # h x 3
    img = np.broadcast_to(col[:, None, :], (h, w, 3)).copy()
    return _np_to_pil(img)


def _spotlights(size: tuple[int, int], n: int = 4, seed: int = 0) -> Image.Image:
    """Soft warm spotlights along the top of the cyc wall, additive blend."""
    w, h = size
    rng = np.random.default_rng(seed)
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    for i in range(n):
        cx = int(w * (i + 0.5) / n) + int(rng.integers(-30, 30))
        cy = int(rng.integers(40, 140))
        r = int(rng.integers(180, 280))
        for ring, brightness in [(r, 60), (int(r * 0.6), 130), (int(r * 0.3), 220)]:
            draw.ellipse(
                (cx - ring, cy - ring, cx + ring, cy + ring),
                fill=brightness,
            )
    return layer.filter(ImageFilter.GaussianBlur(radius=60))


def _vignette_mask(size: tuple[int, int], strength: float = 0.7) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    pad_x, pad_y = w // 5, h // 5
    draw.ellipse((-pad_x, -pad_y, w + pad_x, h + pad_y), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=200))
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    arr = arr * (1 - strength) + strength
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def _cyc_wall(seed: int, base_color: tuple[int, int, int], spotlight: bool = True) -> Image.Image:
    """White/dark cyc wall: solid base + plaster texture + (optional) overhead spotlights."""
    base_arr = np.full((H, W, 3), base_color, dtype=np.float32)

    # Subtle plaster texture
    tex = np.asarray(_smooth_noise((W, H), seed=seed, scale=4), dtype=np.float32)
    tex = (tex - 160.0) * 0.18  # center on 0, low amplitude
    base_arr += tex[..., None]

    if spotlight:
        lights = np.asarray(_spotlights((W, H), n=4, seed=seed + 7), dtype=np.float32)
        # Additive, slightly warm tint
        warm = np.array([1.0, 0.92, 0.78], dtype=np.float32)
        base_arr += lights[..., None] * warm * 0.7

    # Vignette (darker corners pull the eye to the car)
    vig = np.asarray(_vignette_mask((W, H), strength=0.55), dtype=np.float32) / 255.0
    base_arr *= vig[..., None]

    return _np_to_pil(base_arr)


def _floor(top_y: int, color: tuple[int, int, int], pattern: str) -> Image.Image:
    fh = H - top_y
    base_arr = np.full((fh, W, 3), color, dtype=np.float32)

    if pattern == "concrete":
        n = np.asarray(_smooth_noise((W, fh), seed=42, scale=3), dtype=np.float32)
        base_arr += (n - 160.0)[..., None] * 0.22
    elif pattern == "tile":
        floor = _np_to_pil(base_arr)
        draw = ImageDraw.Draw(floor)
        tile_w = 90
        line_color = (max(color[0] - 22, 0), max(color[1] - 22, 0), max(color[2] - 22, 0))
        for x in range(0, W, tile_w):
            draw.line([(x, 0), (x, fh)], fill=line_color, width=2)
        for y in range(0, fh, tile_w):
            draw.line([(0, y), (W, y)], fill=line_color, width=2)
        floor = floor.filter(ImageFilter.GaussianBlur(radius=1.2))
        base_arr = np.asarray(floor, dtype=np.float32)
        n = np.asarray(_smooth_noise((W, fh), seed=11, scale=4), dtype=np.float32)
        base_arr += (n - 160.0)[..., None] * 0.06
    elif pattern == "polished":
        light = np.array([color[0] + 28, color[1] + 28, color[2] + 28], dtype=np.float32)
        dark = np.array(color, dtype=np.float32)
        t = np.linspace(0, 1, fh, dtype=np.float32)[:, None, None]
        col = light * (1 - t) + dark * t
        base_arr = np.broadcast_to(col, (fh, W, 3)).copy()

    floor_img = _np_to_pil(base_arr)

    # Soft shadow band where wall meets floor
    band = Image.new("RGBA", (W, 28), (0, 0, 0, 110))
    band = band.filter(ImageFilter.GaussianBlur(radius=10))
    floor_rgba = floor_img.convert("RGBA")
    floor_rgba.alpha_composite(band, (0, 0))
    return floor_rgba.convert("RGB")


def _composite(wall: Image.Image, floor: Image.Image, top_y: int) -> Image.Image:
    out = wall.copy()
    out.paste(floor, (0, top_y))
    return out


# ---- 5 preset scenes ----------------------------------------------------------

def render_studio_white() -> Image.Image:
    """Bright clean cyc with concrete floor. Closest match to your reference photos."""
    wall = _cyc_wall(seed=1, base_color=(232, 232, 234), spotlight=True)
    floor = _floor(int(H * 0.62), color=(150, 150, 154), pattern="concrete")
    return _composite(wall, floor, int(H * 0.62))


def render_studio_grey_tile() -> Image.Image:
    """Cyc with grey tile floor (matches the rear 3/4 reference shot)."""
    wall = _cyc_wall(seed=2, base_color=(225, 226, 230), spotlight=True)
    floor = _floor(int(H * 0.60), color=(118, 120, 124), pattern="tile")
    return _composite(wall, floor, int(H * 0.60))


def render_studio_charcoal() -> Image.Image:
    """Moody dark cyc, polished floor — premium dealership look."""
    wall = _cyc_wall(seed=3, base_color=(42, 44, 50), spotlight=True)
    floor = _floor(int(H * 0.62), color=(20, 22, 26), pattern="polished")
    return _composite(wall, floor, int(H * 0.62))


def render_studio_warm() -> Image.Image:
    """Warm-toned showroom cyc — luxury feel."""
    wall = _cyc_wall(seed=4, base_color=(232, 222, 208), spotlight=True)
    floor = _floor(int(H * 0.62), color=(120, 110, 98), pattern="concrete")
    return _composite(wall, floor, int(H * 0.62))


def render_studio_blueprint() -> Image.Image:
    """Cool-toned cyc with light blue tint — modern tech look."""
    wall = _cyc_wall(seed=5, base_color=(218, 226, 234), spotlight=True)
    floor = _floor(int(H * 0.62), color=(140, 148, 156), pattern="tile")
    return _composite(wall, floor, int(H * 0.62))


PRESETS = {
    "studio-white": render_studio_white,
    "studio-grey": render_studio_grey_tile,
    "studio-charcoal": render_studio_charcoal,
    "studio-warm": render_studio_warm,
    "studio-blueprint": render_studio_blueprint,
}


def main() -> None:
    for preset_id, fn in PRESETS.items():
        path = ASSETS / f"{preset_id}.jpg"
        print(f"rendering {preset_id} -> {path}")
        img = fn()
        img.save(path, "JPEG", quality=92)
    print(f"\nWrote {len(PRESETS)} backgrounds to {ASSETS}/")


if __name__ == "__main__":
    main()
