from __future__ import annotations

import pytest

from app.storage import get_storage, reset_storage_for_tests


def test_local_storage_is_not_allowed_in_production_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("ALLOW_PUBLIC_LOCAL_STORAGE", "false")
    from app.settings import get_settings

    get_settings.cache_clear()
    reset_storage_for_tests()
    with pytest.raises(RuntimeError, match="Local static storage is public"):
        get_storage()
    get_settings.cache_clear()
    reset_storage_for_tests()


def test_s3_without_public_base_uses_presigned_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setenv("S3_PUBLIC_BASE_URL", "")
    import app.storage as storage_module
    from app.settings import get_settings

    class _Client:
        meta = type("Meta", (), {"endpoint_url": "https://s3.test"})()

        def generate_presigned_url(self, operation, Params, ExpiresIn):  # noqa: N803
            assert operation == "get_object"
            assert Params == {"Bucket": "bucket", "Key": "outputs/job.jpg"}
            assert ExpiresIn == 3600
            return "https://signed.example/outputs/job.jpg"

    class _Boto3:
        @staticmethod
        def client(*_args, **_kwargs):
            return _Client()

    monkeypatch.setitem(__import__("sys").modules, "boto3", _Boto3)
    get_settings.cache_clear()
    reset_storage_for_tests()

    storage = storage_module.get_storage()
    assert storage.public_url("outputs/job.jpg") == "https://signed.example/outputs/job.jpg"
    get_settings.cache_clear()
    reset_storage_for_tests()
