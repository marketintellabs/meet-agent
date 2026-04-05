"""Audio utilities for resampling, chunking, and format conversion."""

from __future__ import annotations

import io
import struct
import wave
from typing import Optional

import numpy as np


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert raw PCM int16 data to WAV format."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def wav_to_pcm(wav_data: bytes) -> tuple[bytes, int, int]:
    """Extract raw PCM data from WAV. Returns (pcm_bytes, sample_rate, channels)."""
    buf = io.BytesIO(wav_data)
    with wave.open(buf, "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()


def float32_to_int16(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array [-1.0, 1.0] to int16 PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


def int16_to_float32(pcm_data: bytes) -> np.ndarray:
    """Convert int16 PCM bytes to float32 numpy array [-1.0, 1.0]."""
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return samples.astype(np.float32) / 32767.0


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling. For production, use librosa or scipy."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


def chunk_audio(
    audio: np.ndarray, chunk_ms: int = 30, sample_rate: int = 16000
) -> list[np.ndarray]:
    """Split audio into fixed-size chunks."""
    chunk_size = int(sample_rate * chunk_ms / 1000)
    return [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]


def create_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Create silent PCM audio of the given duration."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * n_samples


def rms_level(pcm_data: bytes) -> float:
    """Calculate RMS level of int16 PCM audio. Returns value in [0.0, 1.0]."""
    if len(pcm_data) < 2:
        return 0.0
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples**2)) / 32767.0)


def generate_tone(
    frequency: float = 440.0,
    duration_ms: int = 500,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> bytes:
    """Generate a simple sine-wave tone as int16 PCM."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    wave_data = amplitude * np.sin(2 * np.pi * frequency * t)
    return float32_to_int16(wave_data)
