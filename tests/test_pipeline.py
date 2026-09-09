from dataclasses import asdict
import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from genre_cnn.audio import AudioConfig, load_audio, spectrograms
from genre_cnn.predict import Predictor
from genre_cnn.prepare import split_tracks
from genre_cnn.train import train
from web.app import create_app


def wav_bytes(seconds=3, sample_rate=22050, silent=False):
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    signal = np.zeros_like(t) if silent else 0.4 * np.sin(2 * np.pi * 440 * t)
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    return buffer.getvalue()


def test_shared_preprocessing_resamples_bounds_and_pads():
    config = AudioConfig()
    y = load_audio(io.BytesIO(wav_bytes(4, 44100)), config)
    result = spectrograms(y, config)
    assert result.shape == (2, 1, 64, 130)
    assert np.isfinite(result).all()
    assert result.min() >= -1.00001 and result.max() <= 1.00001
    assert len(load_audio(io.BytesIO(wav_bytes(31)))) == 30 * config.sample_rate


@pytest.mark.parametrize(
    "content", [b"not audio", wav_bytes(1), wav_bytes(3, silent=True)]
)
def test_invalid_audio_rejected(content):
    with pytest.raises((ValueError, RuntimeError, OSError)):
        load_audio(io.BytesIO(content))


def test_recording_split_is_reproducible_and_disjoint():
    rows = [
        {"track": f"{genre}/{i}", "genre": genre}
        for genre in ["blues", "rock"]
        for i in range(50)
    ]
    first = split_tracks(rows)
    assert first == split_tracks(rows)
    ids = {split: {r["track"] for r in records} for split, records in first.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert set.union(*ids.values()) == {r["track"] for r in rows}
    assert [len(first[s]) for s in ["train", "validation", "test"]] == [64, 16, 20]
    assert all(
        {r["genre"] for r in records} == {"blues", "rock"} for records in first.values()
    )


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """Exercise optimization/export with synthetic data; these are NOT GTZAN metrics."""
    root = tmp_path_factory.mktemp("synthetic_only")
    cache = root / "cache"
    cache.mkdir()
    rng = np.random.default_rng(42)
    records = []
    for split in ["train", "validation", "test"]:
        for label in range(2):
            name = f"{split}_{label}.npy"
            values = rng.uniform(-1, 1, (2, 1, 64, 130)).astype(np.float16)
            np.save(cache / name, values)
            records.append(
                {
                    "track": name,
                    "genre": ["blues", "rock"][label],
                    "label": label,
                    "cache": name,
                    "segments": 2,
                    "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "split": split,
                }
            )
    manifest = {
        "format_version": 1,
        "seed": 42,
        "config": asdict(AudioConfig()),
        "classes": ["blues", "rock"],
        "tracks": records,
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    metrics = train(
        cache, root / "model", epochs=1, batch_size=2, device="cpu", threads=2
    )
    return root, metrics


def test_training_export_and_inference(synthetic_run):
    root, metrics = synthetic_run
    assert metrics["recordings"] == {"train": 2, "validation": 2, "test": 2}
    assert (
        sum(map(sum, metrics["confusion_matrix"])) == 2
    )  # recordings, not four excerpts
    predictor = Predictor(root / "model/cnn.pt")
    result = predictor.predict(io.BytesIO(wav_bytes(4)))
    assert result["segments"] == 2 and result["seconds_analyzed"] == 4
    assert abs(sum(row["score"] for row in result["scores"]) - 1) < 1e-5
    assert result["genre"] in ["blues", "rock"]
    assert (root / "model/confusion_matrix.png").is_file()


def test_training_rejects_recording_leakage(synthetic_run):
    root, _ = synthetic_run
    manifest = json.loads((root / "cache/manifest.json").read_text())
    manifest["tracks"][-1]["sha256"] = manifest["tracks"][0]["sha256"]
    cache = root / "leaked_cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        train(cache, root / "invalid_model", epochs=1)


def test_api_missing_model_is_unavailable(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status").json()["ready"] is False
        assert client.get("/health").status_code == 503
        assert (
            client.post(
                "/api/predict", files={"file": ("audio.wav", wav_bytes())}
            ).status_code
            == 503
        )


def test_api_real_pipeline_and_invalid_uploads(synthetic_run):
    root, _ = synthetic_run
    with TestClient(create_app(root / "model/cnn.pt")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/status").json()["ready"] is True
        response = client.post(
            "/api/predict", files={"file": ("audio.wav", wav_bytes())}
        )
        assert response.status_code == 200
        assert response.json()["segments"] == 1
        assert (
            client.post(
                "/api/predict", files={"file": ("audio.exe", b"bad")}
            ).status_code
            == 415
        )
        assert (
            client.post(
                "/api/predict", files={"file": ("audio.wav", b"bad")}
            ).status_code
            == 422
        )
        assert client.post("/api/predict").status_code == 422
        # Exercise the body cap cheaply, before the upload can be parsed or decoded.
        import web.app as application

        old_limit = application.MAX_UPLOAD
        application.MAX_UPLOAD = 100
        try:
            assert (
                client.post(
                    "/api/predict", files={"file": ("audio.wav", b"x" * 101)}
                ).status_code
                == 413
            )
            assert (
                client.post(
                    "/api/predict",
                    files={"file": ("audio.wav", b"x" * (1024 * 1024 + 101))},
                ).status_code
                == 413
            )
        finally:
            application.MAX_UPLOAD = old_limit


def test_unsupported_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "untrained.pt"
    torch.save({"format_version": 1, "trained": False}, checkpoint)
    with pytest.raises(ValueError, match="supported trained"):
        Predictor(checkpoint)
