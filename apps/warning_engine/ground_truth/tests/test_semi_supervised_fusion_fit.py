import math
from unittest.mock import AsyncMock

import pytest

from semi_supervised_fusion_fit import self_train, fetch_unlabeled_pairs


def _labeled_rows():
    return [
        {"speech_score": 0.9, "video_score": 0.9, "engaged_label": 1},
        {"speech_score": 0.1, "video_score": 0.1, "engaged_label": 0},
    ] * 10


def test_self_train_terminates_and_returns_finite_weights_with_confident_unlabeled_data():
    unlabeled = [{"speech_score": 0.85, "video_score": 0.88}, {"speech_score": 0.12, "video_score": 0.15}] * 5
    a, b, bias = self_train(_labeled_rows(), unlabeled, confidence_margin=0.3, max_rounds=3)
    assert all(math.isfinite(v) for v in (a, b, bias))


def test_self_train_terminates_when_no_unlabeled_row_is_ever_confident():
    # All unlabeled rows sit right at the decision boundary (score ~0.5) —
    # none should ever cross the 0.3 confidence margin, so the loop must
    # stop at max_rounds rather than spin waiting for confident labels.
    ambiguous = [{"speech_score": 0.5, "video_score": 0.5}] * 10
    a, b, bias = self_train(_labeled_rows(), ambiguous, confidence_margin=0.3, max_rounds=3)
    assert all(math.isfinite(v) for v in (a, b, bias))


@pytest.mark.asyncio
async def test_fetch_unlabeled_pairs_skips_incomplete_rows():
    mock_http = AsyncMock()
    mock_response = AsyncMock()
    # httpx.Response.json()/raise_for_status() are synchronous even on an
    # AsyncClient response, so override AsyncMock's default async behavior
    # for these two attributes to match the real (sync) API.
    mock_response.json = lambda: [
        {"speech_engagement_score": 0.6, "video_attention_score": 0.7},
        {"speech_engagement_score": 0.5, "video_attention_score": None},
        {"speech_engagement_score": None, "video_attention_score": 0.4},
    ]
    mock_response.raise_for_status = lambda: None
    mock_http.get.return_value = mock_response

    pairs = await fetch_unlabeled_pairs("http://backend", [1], mock_http)
    assert pairs == [{"speech_score": 0.6, "video_score": 0.7}]
