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


# ---- v2 floor renderers ------------------------------------------------------
#
# These were added to fix the studio-charcoal / studio-warm / studio-blueprint
# presets, where the original `_floor` patterns produced a flat-looking image
# that was too thin to satisfy the harmonize prompt. The model would then
# "rebuild" the floor and frequently introduce a duplicate-car-shaped
# reflection. The v2 renderers bake in enough specular / texture / grout
# detail that the harmonize pass no longer feels obliged to repaint.


def _polished_floor(
    top_y: int,
    base_color: tuple[int, int, int],
    horizon_lift: int = 60,
    floor_seed: int = 42,
) -> Image.Image:
    """Polished dark floor with a specular band at the wall-floor seam.

    Communicates "polished" via a lightness gradient that peaks just below
    the horizon (where overhead light bounces off the floor and back to
    camera) and falls smoothly into the base colour by mid-floor. No
    car-shaped streaks, so it doesn't read as a literal reflection.
    """
    fh = H - top_y
    base = np.full((fh, W, 3), base_color, dtype=np.float32)

    # Lightness boost from horizon: peaks at ~10% down, dies by ~55% down.
    t = np.linspace(0.0, 1.0, fh, dtype=np.float32)
    spec_curve = np.clip(np.exp(-((t - 0.10) / 0.16) ** 2), 0.0, 1.0)
    base += spec_curve[:, None, None] * float(horizon_lift)

    # Subtle horizontal anisotropy — a real polished floor has a faint
    # cross-direction sheen because the polish lines run with the room.
    cross = np.linspace(0.92, 1.0, W, dtype=np.float32)
    cross_curve = (cross + cross[::-1]) / 2.0  # peaks toward centre
    base *= cross_curve[None, :, None]

    # Soft noise so it isn't perfectly clean.
    n = np.asarray(_smooth_noise((W, fh), seed=floor_seed, scale=8), dtype=np.float32)
    base += (n - 160.0)[..., None] * 0.06

    out = _np_to_pil(base)

    # Strong wall-floor seam shadow.
    band = Image.new("RGBA", (W, 36), (0, 0, 0, 140))
    band = band.filter(ImageFilter.GaussianBlur(radius=14))
    out_rgba = out.convert("RGBA")
    out_rgba.alpha_composite(band, (0, 0))
    return out_rgba.convert("RGB")


def _warm_concrete_floor(
    top_y: int,
    base_color: tuple[int, int, int],
    floor_seed: int = 5,
) -> Image.Image:
    """Sandstone-tinted concrete: warm base + perspective lightening + grain.

    Adds a faint perspective gradient (very slightly lighter near horizon,
    darker toward camera) so the floor has visible depth without looking
    like a polished surface.
    """
    fh = H - top_y
    base = np.full((fh, W, 3), base_color, dtype=np.float32)

    # Perspective gradient: roughly +10% near horizon, -8% near camera.
    t = np.linspace(0.0, 1.0, fh, dtype=np.float32)
    grad = (1.10 - 0.18 * t)
    base *= grad[:, None, None]

    # Concrete grain.
    n = np.asarray(_smooth_noise((W, fh), seed=floor_seed, scale=3), dtype=np.float32)
    base += (n - 160.0)[..., None] * 0.18
    n2 = np.asarray(_smooth_noise((W, fh), seed=floor_seed + 9, scale=10), dtype=np.float32)
    base += (n2 - 160.0)[..., None] * 0.06

    out = _np_to_pil(base)

    # Gentle seam shadow.
    band = Image.new("RGBA", (W, 30), (0, 0, 0, 110))
    band = band.filter(ImageFilter.GaussianBlur(radius=10))
    out_rgba = out.convert("RGBA")
    out_rgba.alpha_composite(band, (0, 0))
    return out_rgba.convert("RGB")


def _blue_tile_floor(
    top_y: int,
    tile_color: tuple[int, int, int],
    grout_color: tuple[int, int, int],
    tile_w: int = 110,
) -> Image.Image:
    """Cool light tile floor with proper grout lines and a subtle perspective tint.

    The grout colour is set by the caller (slightly darker than the tile)
    so the grid reads clearly without looking heavy. A tiny vertical
    gradient gives the floor depth.
    """
    fh = H - top_y
    base = np.full((fh, W, 3), tile_color, dtype=np.float32)

    # Vertical perspective: +5% near horizon -> -5% near camera.
    t = np.linspace(0.0, 1.0, fh, dtype=np.float32)
    grad = (1.05 - 0.10 * t)
    base *= grad[:, None, None]

    # Faint surface noise so the tiles aren't dead flat.
    n = np.asarray(_smooth_noise((W, fh), seed=23, scale=12), dtype=np.float32)
    base += (n - 160.0)[..., None] * 0.05

    floor = _np_to_pil(base)

    # Tile grid.
    draw = ImageDraw.Draw(floor)
    for x in range(0, W, tile_w):
        draw.line([(x, 0), (x, fh)], fill=grout_color, width=2)
    for y in range(0, fh, tile_w):
        draw.line([(0, y), (W, y)], fill=grout_color, width=2)
    floor = floor.filter(ImageFilter.GaussianBlur(radius=1.0))

    # Seam shadow.
    band = Image.new("RGBA", (W, 28), (0, 0, 0, 105))
    band = band.filter(ImageFilter.GaussianBlur(radius=10))
    out_rgba = floor.convert("RGBA")
    out_rgba.alpha_composite(band, (0, 0))
    return out_rgba.convert("RGB")


def _warm_overhead_glow(seed: int) -> Image.Image:
    """A single soft warm halo high on the wall (replaces multi-spotlight
    pattern for the warm preset). One concentrated source reads more like
    a real showroom than four scattered ones.
    """
    layer = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(layer)
    rng = np.random.default_rng(seed)
    cx = W // 2 + int(rng.integers(-40, 40))
    cy = int(H * 0.16)
    for r, brightness in [(900, 30), (550, 80), (280, 170)]:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=brightness)
    return layer.filter(ImageFilter.GaussianBlur(radius=80))


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
    """Moody dark cyc, polished floor with a horizon specular band.

    The wall stays low-key (no spotlights) so the floor's specular band
    becomes the visual focus. The polished_floor renderer bakes in the
    "polished" cue with a horizon lightness peak rather than a flat
    uniform gradient — so Qwen has no reason to invent a literal car
    reflection on the floor.
    """
    wall = _cyc_wall(seed=3, base_color=(38, 40, 46), spotlight=False)
    # Add a subtle warm rim along the very top of the wall so the cyc has
    # depth without bright spotlights leaking onto the floor.
    rim_layer = Image.new("L", (W, H), 0)
    rim_draw = ImageDraw.Draw(rim_layer)
    rim_draw.rectangle((0, 0, W, int(H * 0.18)), fill=80)
    rim_layer = rim_layer.filter(ImageFilter.GaussianBlur(radius=120))
    rim_arr = np.asarray(rim_layer, dtype=np.float32)
    wall_arr = np.asarray(wall, dtype=np.float32)
    warm = np.array([1.0, 0.92, 0.80], dtype=np.float32)
    wall_arr += rim_arr[..., None] * warm * 0.35
    wall = _np_to_pil(wall_arr)

    floor_y = int(H * 0.62)
    floor = _polished_floor(
        floor_y, base_color=(20, 22, 28), horizon_lift=58, floor_seed=42,
    )
    return _composite(wall, floor, floor_y)


def render_studio_warm() -> Image.Image:
    """Warm-toned showroom: cream cyc + sandstone concrete, single warm halo.

    Uses _warm_overhead_glow for one concentrated warm light source
    instead of multiple cool spotlights, and the new sandstone
    concrete renderer with perspective gradient.
    """
    wall = _cyc_wall(seed=4, base_color=(232, 222, 208), spotlight=False)
    glow = np.asarray(_warm_overhead_glow(seed=4), dtype=np.float32)
    wall_arr = np.asarray(wall, dtype=np.float32)
    warm = np.array([1.0, 0.86, 0.66], dtype=np.float32)
    wall_arr += glow[..., None] * warm * 0.65
    wall = _np_to_pil(wall_arr)

    floor_y = int(H * 0.62)
    floor = _warm_concrete_floor(
        floor_y, base_color=(150, 134, 112), floor_seed=8,
    )
    return _composite(wall, floor, floor_y)


def render_studio_blueprint() -> Image.Image:
    """Cool-toned cyc with a noticeable blue tint and a real blue tile floor.

    Bumps the cyc's blue from a barely-perceptible 7-point shift to
    something that actually reads as cool, and gives the floor a
    coherent tile grid with darker grout that matches the prompt.
    """
    wall = _cyc_wall(seed=5, base_color=(196, 212, 230), spotlight=True)
    # Push the wall a touch cooler still by lifting blue and gently
    # darkening warm channels.
    wall_arr = np.asarray(wall, dtype=np.float32)
    cool = np.array([0.96, 0.99, 1.05], dtype=np.float32)
    wall_arr *= cool
    wall = _np_to_pil(wall_arr)

    floor_y = int(H * 0.62)
    floor = _blue_tile_floor(
        floor_y,
        tile_color=(176, 188, 202),
        grout_color=(126, 138, 152),
        tile_w=110,
    )
    return _composite(wall, floor, floor_y)


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
