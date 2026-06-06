"""Talor SERP engine constants and utilities."""

DEFAULT_ENGINE = "google"

SUPPORTED_ENGINES = [
    # Google
    "google",
    "google_finance",
    "google_finance_markets",
    "google_flights",
    "google_hotels",
    "google_images",
    "google_jobs",
    "google_lens",
    "google_local",
    "google_maps",
    "google_news",
    "google_patents",
    "google_patents_details",
    "google_play",
    "google_play_books",
    "google_play_games",
    "google_play_movies",
    "google_play_product",
    "google_scholar",
    "google_scholar_author",
    "google_scholar_cite",
    "google_shopping",
    "google_trends",
    "google_videos",
    "google_web",
    # Bing
    "bing",
    "bing_images",
    "bing_maps",
    "bing_news",
    "bing_shopping",
    "bing_videos",
    # Yandex
    "yandex",
    # DuckDuckGo
    "duckduckgo",
]

ENGINE_CATEGORIES = {
    "google": [e for e in SUPPORTED_ENGINES if e.startswith("google")],
    "bing": [e for e in SUPPORTED_ENGINES if e.startswith("bing")],
    "yandex": ["yandex"],
    "duckduckgo": ["duckduckgo"],
}
