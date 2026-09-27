"""Offline window scoring — reproduces production `semantic_analysis` scoring.

Embeds the contract topic once (via `topic_text(topic, scope)`) and each
`ReplayWindow`'s cleaned `.text`, computes cosine similarity exactly the way
`apps/semantic_analysis/main.py` does (`SentenceTransformer.encode(...,
convert_to_tensor=True)` + `sentence_transformers.util.cos_sim` + `float(...
.item())`), and classifies with the shared production `classify` thresholds.

This module does scoring only — no windowing, no mechanism (RHO/thresholds
beyond the shared `classify`), no HTTP.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

_WINDOWING_PATH = (
    Path(__file__).resolve().parents[2] / "apps" / "semantic_analysis" / "windowing.py"
)
_spec = importlib.util.spec_from_file_location("_embed_windowing", _WINDOWING_PATH)
_windowing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_windowing)

classify = _windowing.classify
topic_text = _windowing.topic_text

_MODEL_NAME = "all-MiniLM-L6-v2"
_cached_model = None


@dataclass(frozen=True)
class WindowScore:
    index: int
    similarity: float
    classification: str


def _get_cached_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer(_MODEL_NAME)
    return _cached_model


def score_windows(windows, topic: str, scope: str, model=None) -> list:
    """Score each window's cleaned text against the contract topic.

    `model=None` loads and caches `SentenceTransformer("all-MiniLM-L6-v2")`
    at module level (shared across calls). Passing an explicit `model`
    never populates or consumes that cache.
    """
    if not windows:
        return []

    active_model = model if model is not None else _get_cached_model()

    topic_embedding = active_model.encode(topic_text(topic, scope), convert_to_tensor=True)

    scores = []
    for window in windows:
        window_embedding = active_model.encode(window.text, convert_to_tensor=True)
        similarity = float(util.cos_sim(topic_embedding, window_embedding).item())
        scores.append(
            WindowScore(
                index=window.index,
                similarity=similarity,
                classification=classify(similarity),
            )
        )
    return scores
