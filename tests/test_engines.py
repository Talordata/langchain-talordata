"""Tests for langchain_talordata.engines module."""

from langchain_talordata.engines import (
    DEFAULT_ENGINE,
    ENGINE_CATEGORIES,
    SUPPORTED_ENGINES,
)


class TestEngineConstants:
    def test_default_engine(self):
        assert DEFAULT_ENGINE == "google"

    def test_supported_engines_is_list(self):
        assert isinstance(SUPPORTED_ENGINES, list)
        assert len(SUPPORTED_ENGINES) > 0

    def test_all_google_engines_present(self):
        google_engines = [e for e in SUPPORTED_ENGINES if e.startswith("google")]
        assert len(google_engines) >= 20

    def test_all_bing_engines_present(self):
        bing_engines = [e for e in SUPPORTED_ENGINES if e.startswith("bing")]
        assert len(bing_engines) >= 5

    def test_yandex_present(self):
        assert "yandex" in SUPPORTED_ENGINES

    def test_duckduckgo_present(self):
        assert "duckduckgo" in SUPPORTED_ENGINES

    def test_no_duplicates(self):
        assert len(SUPPORTED_ENGINES) == len(set(SUPPORTED_ENGINES))

    def test_engine_categories_keys(self):
        assert "google" in ENGINE_CATEGORIES
        assert "bing" in ENGINE_CATEGORIES
        assert "yandex" in ENGINE_CATEGORIES
        assert "duckduckgo" in ENGINE_CATEGORIES

    def test_engine_categories_cover_all(self):
        all_categorized = []
        for engines in ENGINE_CATEGORIES.values():
            all_categorized.extend(engines)
        assert sorted(all_categorized) == sorted(SUPPORTED_ENGINES)
