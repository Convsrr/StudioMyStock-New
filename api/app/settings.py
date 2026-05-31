"""Typed application settings loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    public_base_url: str = "http://localhost:8000"

    # Database
    database_url: str = "sqlite+aiosqlite:///./storage/app.db"

    # Redis / queue
    redis_url: str = "redis://localhost:6379/0"

    # Replicate
    replicate_api_token: str = ""
    use_replicate: bool = True
    replicate_bg_remover: str = (
        "851-labs/background-remover:"
        "a029dff38972b5fda4ec5d75d7d1cd25aeff621d2cf4946a41055d7db66b80bc"
    )
    replicate_relight: str = (
        "zsxkib/ic-light:"
        "0c184710f902c2dd4593e1aa5e93fcd8f937c2e3d96d0e9ea0f22a87e3f0e7d3"
    )
    replicate_upscaler: str = (
        "nightmareai/real-esrgan:"
        "f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"
    )
    replicate_plate_detector: str = ""

    # Picsart
    picsart_api_key: str = ""

    # Segmentation provider: auto picks Picsart if configured, then Replicate.
    segmentation_provider: Literal["auto", "picsart", "replicate", "none"] = "auto"

    # Storage
    storage_backend: Literal["local", "s3"] = "local"
    storage_dir: Path = Path("./storage")
    s3_bucket: str = ""
    s3_region: str = "auto"
    s3_endpoint_url: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_public_base_url: str = ""
    s3_presigned_url_ttl_seconds: int = 3600
    allow_public_local_storage: bool = False

    # API auth + rate limit
    api_keys: str = ""
    rate_limit_per_minute: int = 30

    # Pipeline tuning
    max_input_pixels: int = 24_000_000
    working_max_dim: int = 2048
    output_max_dim: int = 3840
    jpeg_quality: int = Field(default=92, ge=60, le=100)
    job_timeout_seconds: int = 300

    # Pre-segmentation prep (denoise + auto WB + highlight recovery)
    enable_prep: bool = True
    prep_denoise: bool = True
    prep_white_balance: bool = True
    prep_highlight_recover: bool = True
    prep_wb_strength: float = Field(default=0.6, ge=0.0, le=1.0)

    # Skip segment/compose/shadow when input already looks like a studio shot
    enable_studio_shortcircuit: bool = True
    # Production default is false because passthrough ignores the chosen
    # background. Keep it as an explicit opt-in for demos or existing-studio
    # polishing flows.
    allow_studio_shortcircuit_passthrough: bool = False

    # Reflection stage (deterministic floor reflection under the car)
    enable_reflection: bool = True
    reflection_opacity: float = Field(default=0.18, ge=0.0, le=1.0)
    reflection_blur: int = Field(default=10, ge=0, le=64)
    reflection_floor_tint: float = Field(default=0.55, ge=0.0, le=1.0)

    # Chassis ambient occlusion (soft darkening under the car). Lives in
    # the reflection stage so it shares the contact-line anchor.
    enable_chassis_ao: bool = True
    chassis_ao_strength: float = Field(default=0.45, ge=0.0, le=1.0)

    # Quality guard: detect AI-introduced duplicate cars and fall back
    # to the deterministic composite when triggered.
    enable_duplicate_guard: bool = True

    # preserve_car: how strongly to keep the original cutout pixels vs
    # Qwen's lit version of the car region. 0.92 = 92% original, 8% AI.
    car_identity_strength: float = Field(default=0.92, ge=0.5, le=1.0)

    # When True, preserve_car classifies chrome / glass / paint inside
    # the cutout and lets more of Qwen's harmonized lighting through on
    # chrome and glass while keeping plates and badges at full identity.
    preserve_car_adaptive: bool = True

    # Run pipeline inline via FastAPI BackgroundTasks instead of enqueuing to Redis.
    # Useful for local dev or single-host deployments without a worker process.
    inline_processing: bool = False

    @field_validator("storage_dir", mode="before")
    @classmethod
    def _resolve_storage_dir(cls, v: object) -> Path:
        return Path(str(v)).resolve()

    @property
    def replicate_enabled(self) -> bool:
        return self.use_replicate and bool(self.replicate_api_token)

    @property
    def picsart_enabled(self) -> bool:
        return bool(self.picsart_api_key)

    @property
    def active_segmentation_provider(self) -> str:
        """Resolved segmentation provider: 'picsart', 'replicate', or 'none'."""
        if self.segmentation_provider == "auto":
            if self.picsart_enabled:
                return "picsart"
            if self.replicate_enabled:
                return "replicate"
            return "none"
        return self.segmentation_provider

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.storage_dir / "outputs"

    @property
    def cache_dir(self) -> Path:
        return self.storage_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for d in (settings.storage_dir, settings.uploads_dir, settings.outputs_dir, settings.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return settings
