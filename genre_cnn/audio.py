"""One shared preprocessing path for training and inference."""

from dataclasses import dataclass
import math
import numpy as np
import librosa
import soundfile as sf

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    seconds: float = 3.0
    max_seconds: float = 30.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    top_db: float = 80.0

    @property
    def samples(self):
        return round(self.sample_rate * self.seconds)

    @property
    def frames(self):
        return 1 + self.samples // self.hop_length


def load_audio(source, config=AudioConfig()):
    """Read at most 30 seconds and reject malformed/too-short uploads."""
    with sf.SoundFile(source) as audio:
        if not 8000 <= audio.samplerate <= 192000 or not 1 <= audio.channels <= 8:
            raise ValueError("Unsupported sample rate or number of channels")
        if audio.frames / audio.samplerate < config.seconds:
            raise ValueError(
                f"Please provide at least {config.seconds:g} seconds of audio"
            )
        y = audio.read(
            frames=int(audio.samplerate * config.max_seconds),
            dtype="float32",
            always_2d=True,
        ).mean(axis=1)
        source_rate = audio.samplerate
    if not np.isfinite(y).all():
        raise ValueError("Audio contains invalid sample values")
    if source_rate != config.sample_rate:
        y = librosa.resample(y, orig_sr=source_rate, target_sr=config.sample_rate)
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError("The audio is silent. Please choose a music clip")
    return np.asarray(y, dtype=np.float32)


def segments(y, config=AudioConfig()):
    y = y[: round(config.sample_rate * config.max_seconds)]
    count = max(1, math.ceil(len(y) / config.samples))
    result = np.zeros((count, config.samples), dtype=np.float32)
    for i in range(count):
        part = y[i * config.samples : (i + 1) * config.samples]
        result[i, : len(part)] = part
    return result


def spectrograms(y, config=AudioConfig()):
    result = []
    for chunk in segments(y, config):
        power = librosa.feature.melspectrogram(
            y=chunk,
            sr=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            fmin=20,
            fmax=config.sample_rate / 2,
            power=2.0,
        )
        # A fixed [-80, 0] dB to [-1, 1] transform needs no fitted test statistics.
        if float(power.max()) < 1e-12:
            normalized = np.full_like(power, -1, dtype=np.float32)
        else:
            db = librosa.power_to_db(power, ref=np.max, top_db=config.top_db)
            normalized = 2 * np.clip(db, -config.top_db, 0) / config.top_db + 1
        result.append(normalized[:, : config.frames])
    return np.asarray(result, dtype=np.float32)[:, None, :, :]
