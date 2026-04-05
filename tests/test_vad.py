"""Tests for Voice Activity Detection."""

import numpy as np

from meet_agent.audio import float32_to_int16
from meet_agent.pipeline.vad import VADProcessor


def _make_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * n


def _make_noise(duration_ms: int, sample_rate: int = 16000, amplitude: float = 0.3) -> bytes:
    n = int(sample_rate * duration_ms / 1000)
    audio = amplitude * np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate).astype(np.float32)
    return float32_to_int16(audio)


def test_vad_detects_silence():
    vad = VADProcessor(threshold=0.5, min_speech_ms=100, min_silence_ms=200)
    silence = _make_silence(30)
    result = vad.process_chunk(silence)
    assert result is None


def test_vad_detects_speech_segment():
    vad = VADProcessor(threshold=0.5, min_speech_ms=100, min_silence_ms=200, sample_rate=16000)
    # Feed speech chunks
    for _ in range(20):
        vad.process_chunk(_make_noise(30))

    # Feed silence to trigger segment end
    result = None
    for _ in range(30):
        r = vad.process_chunk(_make_silence(30))
        if r is not None:
            result = r
            break

    assert result is not None
    assert len(result) > 0


def test_vad_flush():
    vad = VADProcessor(min_speech_ms=100, min_silence_ms=500)
    for _ in range(20):
        vad.process_chunk(_make_noise(30))

    result = vad.flush()
    assert result is not None
    assert len(result) > 0
