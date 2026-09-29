import pytest


def _window(window_id, classification="incorrect", **extra):
    body = {
        "barter_id": 1, "window_id": window_id, "classification": classification,
        "similarity_score": 0.5, "text_preview": "I will not pay you.",
        "timestamp_start": 0.0, "timestamp_end": 25.0,
    }
    body.update(extra)
    return body


def _calls(mock_http, suffix):
    return [c for c in mock_http.post.call_args_list if c.args[0].endswith(suffix)]


@pytest.mark.asyncio
async def test_reversal_flag_forwarded_to_backend(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/window/result", json=_window(1, meaning_reversal_detected=True))
    assert resp.status_code == 200

    result_calls = _calls(mock_http, "/window/result")
    assert len(result_calls) == 1
    assert result_calls[0].kwargs["json"]["meaning_reversal_detected"] is True


@pytest.mark.asyncio
async def test_reversal_flag_defaults_false_when_omitted(warning_client):
    client, mock_http, we_main = warning_client
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})

    resp = await client.post("/window/result", json=_window(1, classification="correct"))
    assert resp.status_code == 200

    result_calls = _calls(mock_http, "/window/result")
    assert result_calls[0].kwargs["json"]["meaning_reversal_detected"] is False


async def _run_sequence(client, mock_http, **extra):
    mock_http.post.reset_mock()
    await client.post("/session/1/init", json={"teacher_user_id": 1, "learner_user_id": 2})
    responses = []
    for window_id in (1, 2, 3):
        resp = await client.post("/window/result", json=_window(window_id, **extra))
        responses.append(resp.json())
    # Drop the wall-clock timestamp, the only field expected to differ between runs.
    warnings = [
        {k: v for k, v in c.kwargs["json"].items() if k != "timestamp"}
        for c in _calls(mock_http, "/warnings/log")
    ]
    return responses, warnings


@pytest.mark.asyncio
async def test_reversal_flag_does_not_change_escalation(warning_client):
    client, mock_http, we_main = warning_client

    base_resp, base_warnings = await _run_sequence(client, mock_http)
    we_main.sessions.clear()
    flag_resp, flag_warnings = await _run_sequence(
        client, mock_http, meaning_reversal_detected=True
    )

    assert len(base_warnings) == 2
    assert flag_resp == base_resp
    assert flag_warnings == base_warnings
    assert flag_warnings[-1]["severity"] == "severe"
