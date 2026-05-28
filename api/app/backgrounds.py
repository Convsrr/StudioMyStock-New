"""Background catalog.

Production: serves real PNG/JPEG assets rendered by ``scripts/render_backgrounds.py``.
Each preset has both an ``image`` (used for compositing fallback) and a ``prompt``
(used by the Qwen Image Edit harmonize stage to describe the target scene).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .settings import get_settings

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "backgrounds"


@dataclass(frozen=True)
class BackgroundPreset:
    id: str
    name: str
    description: str
    prompt: str  # Used by the harmonize stage as the target-scene description
    floor_y_ratio: float  # Where the floor line sits, as a fraction of canvas height
    light_direction: str  # "top", "top-left", "top-right" — for shadow synthesis


PRESETS: list[BackgroundPreset] = [
    BackgroundPreset(
        id="studio-white",
        name="Studio White",
        description="Clean white cyc wall, light concrete floor",
        prompt=(
            "professional automotive photography studio, seamless white cyc wall "
            "with subtle warm overhead spotlights, light grey concrete floor, "
            "soft even lighting, no other objects, magazine-quality, photorealistic"
        ),
        floor_y_ratio=0.62,
        light_direction="top",
    ),
    BackgroundPreset(
        id="studio-grey",
        name="Studio Grey",
        description="Light cyc, mid-grey tile floor",
        prompt=(
            "professional automotive photography studio, seamless light grey cyc wall, "
            "polished grey square tile floor, soft overhead studio lighting, "
            "no other objects, dealership style, photorealistic"
        ),
        floor_y_ratio=0.60,
        light_direction="top",
    ),
    BackgroundPreset(
        id="studio-charcoal",
        name="Studio Charcoal",
        description="Moody dark cyc, polished black floor",
        prompt=(
            "premium automotive photography studio, dark charcoal seamless cyc wall, "
            "glossy polished black floor reflecting subtle highlights, "
            "moody low-key lighting with rim lights, no other objects, "
            "luxury car magazine cover, photorealistic"
        ),
        floor_y_ratio=0.62,
        light_direction="top-left",
    ),
    BackgroundPreset(
        id="studio-warm",
        name="Studio Warm",
        description="Warm cream cyc, sandstone floor",
        prompt=(
            "warm-toned automotive showroom, cream-coloured seamless cyc wall, "
            "warm sandstone concrete floor, soft warm overhead lighting, "
            "no other objects, premium dealership feel, photorealistic"
        ),
        floor_y_ratio=0.62,
        light_direction="top-right",
    ),
    BackgroundPreset(
        id="studio-blueprint",
        name="Studio Blueprint",
        description="Cool blue-tinted cyc, light tile floor",
        prompt=(
            "modern automotive photography studio, cool blue-grey seamless cyc wall, "
            "light grey tile floor, clean overhead lighting with subtle cool tint, "
            "no other objects, modern tech feel, photorealistic"
        ),
        floor_y_ratio=0.62,
        light_direction="top",
    ),
]

_BY_ID = {p.id: p for p in PRESETS}


def list_presets() -> list[dict]:
    return [
        {"id": p.id, "name": p.name, "description": p.description}
        for p in PRESETS
    ]


def is_valid(preset_id: str) -> bool:
    return preset_id in _BY_ID


def get_preset(preset_id: str) -> BackgroundPreset:
    if preset_id not in _BY_ID:
        raise KeyError(f"Unknown background: {preset_id}")
    return _BY_ID[preset_id]


def get_background(preset_id: str, size: tuple[int, int]) -> Image.Image:
    """Return the background image at the requested size, with on-disk caching.

    The shipped preset assets are 1920x1280 landscape. When the requested
    size has a different aspect ratio (square or portrait), we centre-crop
    the source asset to the target aspect and then resize, instead of
    naively stretching. The studio scenes are roughly translation-symmetric
    horizontally so a centre slice still reads as a believable studio.
    The floor-line ratio is preserved by the proportional resize.
    """
    if preset_id not in _BY_ID:
        raise KeyError(f"Unknown background: {preset_id}")

    cache_dir = get_settings().cache_dir / "backgrounds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{preset_id}_{size[0]}x{size[1]}.jpg"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    src = ASSETS_DIR / f"{preset_id}.jpg"
    if not src.exists():
        raise FileNotFoundError(
            f"Background asset missing: {src}. Run scripts/render_backgrounds.py."
        )
    img = Image.open(src).convert("RGB")
    if img.size != size:
        img = _reframe_to_aspect(img, size)
    img.save(cache_path, "JPEG", quality=92)
    return img


def _reframe_to_aspect(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Centre-crop ``img`` to the aspect of ``size``, then resize to ``size``.

    If the source already has the target aspect (within 1%), this is a
    plain resize.
    """
    target_w, target_h = size
    src_w, src_h = img.size
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h

    if abs(target_aspect - src_aspect) < 0.01:
        return img.resize(size, Image.LANCZOS)

    if target_aspect > src_aspect:
        # Target is wider than source: crop the source vertically.
        crop_h = int(round(src_w / target_aspect))
        crop_h = min(crop_h, src_h)
        offset = (src_h - crop_h) // 2
        cropped = img.crop((0, offset, src_w, offset + crop_h))
    else:
        # Target is narrower than source (e.g. portrait canvas from a
        # landscape asset): crop the source horizontally.
        crop_w = int(round(src_h * target_aspect))
        crop_w = min(crop_w, src_w)
        offset = (src_w - crop_w) // 2
        cropped = img.crop((offset, 0, offset + crop_w, src_h))

    return cropped.resize(size, Image.LANCZOS)
