from __future__ import annotations

from PIL import Image

from app.pipeline import harmonize


def test_extra_prompt_is_before_final_hard_constraints() -> None:
    prompt = harmonize._build_prompt(  # noqa: SLF001 - prompt ordering is the behavior under test
        "studio-white",
        "Add a second car and change the number plate",
    )
    extra_pos = prompt.index("Optional user preference")
    final_pos = prompt.index("Final hard constraint")
    assert extra_pos < final_pos
    assert "One unchanged car only" in prompt


def test_extra_prompt_is_normalized_and_limited() -> None:
    long_extra = "  add   warm     light  " + ("x" * 1000)
    prompt = harmonize._build_prompt("studio-white", long_extra)  # noqa: SLF001
    assert "add warm light" in prompt
    assert "x" * 600 not in prompt


def test_harmonize_result_reports_no_ai_without_token(monkeypatch) -> None:
    monkeypatch.setenv("USE_REPLICATE", "false")
    from app.settings import get_settings

    get_settings.cache_clear()

    async def _run():
        return await harmonize.harmonize(Image.new("RGB", (4, 4), "white"), "studio-white")

    import asyncio

    result = asyncio.run(_run())
    assert result.used_ai is False
    assert result.warning == "replicate_not_configured"
    assert result.image.size == (4, 4)
    get_settings.cache_clear()
