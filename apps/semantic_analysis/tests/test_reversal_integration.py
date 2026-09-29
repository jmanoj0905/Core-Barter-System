import pytest

from windowing import LOWER, clean_text, classify


def _segment(text):
    # user_id 1 is the teacher in the fixture contract for barter 1; 30s
    # >= WINDOW_DURATION_THRESHOLD so the window is processed immediately.
    return {
        "barter_id": 1,
        "user_id": 1,
        "text": text,
        "duration_seconds": 30.0,
        "timestamp_start": 0.0,
        "timestamp_end": 30.0,
    }


def _window_payload(mock_http):
    calls = [c for c in mock_http.post.call_args_list if c.args[0].endswith("/window/result")]
    assert len(calls) == 1
    return calls[0].kwargs["json"]


def _cosine_class(sa_main, text):
    cleaned = clean_text(text)
    sim = sa_main.cosine_sim(sa_main.embed(cleaned), sa_main.contracts[1]["topic_embedding"])
    return sim, classify(sim)


@pytest.mark.asyncio
async def test_reversal_forces_incorrect_and_sets_flag(semantic_client):
    client, mock_http, sa_main = semantic_client
    text = "I will not pay you the money for the agreed car repair."
    sa_main.contracts[1]["topic_embedding"] = sa_main.embed(
        "paying money for a car repair agreement"
    )

    # Precondition: cosine alone would NOT have said incorrect.
    sim, cosine_class = _cosine_class(sa_main, text)
    assert sim >= LOWER and cosine_class != "incorrect", (sim, cosine_class)

    resp = await client.post("/ingest/segment", json=_segment(text))
    assert resp.status_code == 200

    payload = _window_payload(mock_http)
    assert payload["classification"] == "incorrect"
    assert payload["meaning_reversal_detected"] is True


@pytest.mark.asyncio
async def test_no_reversal_leaves_flag_false(semantic_client):
    client, mock_http, sa_main = semantic_client
    text = "I will pay you the money for the agreed car repair."
    sa_main.contracts[1]["topic_embedding"] = sa_main.embed(
        "paying money for a car repair agreement"
    )
    _, cosine_class = _cosine_class(sa_main, text)

    resp = await client.post("/ingest/segment", json=_segment(text))
    assert resp.status_code == 200

    payload = _window_payload(mock_http)
    assert payload["meaning_reversal_detected"] is False
    # No override: classification is exactly what cosine says.
    assert payload["classification"] == cosine_class


@pytest.mark.asyncio
async def test_reversal_override_idempotent_when_already_incorrect_by_cosine(semantic_client):
    client, mock_http, sa_main = semantic_client
    text = "I will not pay you the money for the car repair."
    sa_main.contracts[1]["topic_embedding"] = sa_main.embed(
        "a completely unrelated topic about gardening"
    )

    # Precondition: cosine alone already says incorrect.
    sim, cosine_class = _cosine_class(sa_main, text)
    assert cosine_class == "incorrect", (sim, cosine_class)

    resp = await client.post("/ingest/segment", json=_segment(text))
    assert resp.status_code == 200

    payload = _window_payload(mock_http)
    assert payload["classification"] == "incorrect"
    assert payload["meaning_reversal_detected"] is True
