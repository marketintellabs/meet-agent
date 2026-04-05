"""Tests for audio utilities."""

import numpy as np

from meet_agent.audio import (
    chunk_audio,
    create_silence,
    float32_to_int16,
    generate_tone,
    int16_to_float32,
    pcm_to_wav,
    resample,
    rms_level,
    wav_to_pcm,
)


def test_pcm_wav_roundtrip():
    original = generate_tone(440, 500, 16000)
    wav = pcm_to_wav(original, 16000)
    pcm, sr, ch = wav_to_pcm(wav)
    assert sr == 16000
    assert ch == 1
    assert pcm == original


def test_float32_int16_roundtrip():
    original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm = float32_to_int16(original)
    recovered = int16_to_float32(pcm)
    np.testing.assert_allclose(recovered, original, atol=1e-4)


def test_resample_same_rate():
    audio = np.ones(1600, dtype=np.float32)
    result = resample(audio, 16000, 16000)
    assert len(result) == 1600


def test_resample_downsample():
    audio = np.ones(16000, dtype=np.float32)
    result = resample(audio, 16000, 8000)
    assert len(result) == 8000


def test_chunk_audio():
    audio = np.zeros(4800, dtype=np.float32)
    chunks = chunk_audio(audio, chunk_ms=30, sample_rate=16000)
    assert len(chunks) == 10
    assert len(chunks[0]) == 480


def test_create_silence():
    silence = create_silence(100, 16000)
    assert len(silence) == 3200  # 1600 samples * 2 bytes


def test_rms_level_silence():
    silence = create_silence(100, 16000)
    assert rms_level(silence) == 0.0


def test_rms_level_tone():
    tone = generate_tone(440, 100, 16000, amplitude=0.5)
    level = rms_level(tone)
    assert 0.2 < level < 0.6


def test_generate_tone():
    tone = generate_tone(440, 500, 16000)
    assert len(tone) == 16000  # 8000 samples * 2 bytes
