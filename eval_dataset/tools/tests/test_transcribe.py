import json
from unittest.mock import MagicMock, patch
from eval_dataset.tools.transcribe import transcribe_audio


def test_transcribe_audio_uploads_polls_downloads_and_cleans_up(tmp_path):
    wav_path = tmp_path / "sess_A01.wav"
    wav_path.write_bytes(b"fake-wav-bytes")

    s3_client = MagicMock()
    transcribe_client = MagicMock()
    transcribe_client.get_transcription_job.side_effect = [
        {"TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}},
        {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {"TranscriptFileUri": "https://fake-uri/transcript.json"},
            }
        },
    ]

    aws_transcript_payload = {
        "results": {
            "items": [
                {"type": "pronunciation", "alternatives": [{"content": "Hello"}], "start_time": "0.0", "end_time": "0.4"},
                {"type": "punctuation", "alternatives": [{"content": "."}]},
                {"type": "pronunciation", "alternatives": [{"content": "there"}], "start_time": "0.5", "end_time": "0.9"},
            ]
        }
    }

    with patch("eval_dataset.tools.transcribe._download_json", return_value=aws_transcript_payload), \
         patch("eval_dataset.tools.transcribe.time.sleep"):
        result = transcribe_audio(s3_client, transcribe_client, str(wav_path), bucket="test-bucket", job_name="job-1")

    s3_client.upload_file.assert_called_once_with(str(wav_path), "test-bucket", "job-1.wav")
    transcribe_client.start_transcription_job.assert_called_once()
    s3_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="job-1.wav")

    assert result == {"words": [{"text": "Hello", "start": 0.0, "end": 0.4}, {"text": "there", "start": 0.5, "end": 0.9}]}
