"""Voice Activity Detection using silero-vad."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VADProcessor:
    """Detects speech segments in a stream of PCM audio chunks.

    Uses silero-vad when available, falls back to simple energy-based detection.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
        pre_speech_pad_ms: int = 300,
        sample_rate: int = 16000,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.pre_speech_pad_ms = pre_speech_pad_ms
        self.sample_rate = sample_rate

        self._is_speaking = False
        self._speech_buffer: list[bytes] = []
        self._silence_ms = 0
        self._speech_ms = 0

        # Ring buffer for pre-speech padding
        pad_chunks = max(1, int(pre_speech_pad_ms / 30))
        self._pre_buffer: deque[bytes] = deque(maxlen=pad_chunks)

        # Try to load silero-vad
        self._silero_model = None
        self._use_silero = False
        try:
            import torch

            model, utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
            self._silero_model = model
            self._get_speech_prob = utils[0] if callable(utils[0]) else None
            self._use_silero = True
            logger.info("VAD: using silero-vad")
        except Exception:
            logger.info("VAD: silero-vad not available, using energy-based detection")

    def process_chunk(self, pcm_chunk: bytes) -> Optional[bytes]:
        """Process a PCM int16 chunk (~30ms). Returns a complete speech segment or None.

        When speech ends (after min_silence_ms of silence), returns the full
        concatenated speech segment including pre-speech padding.
        """
        chunk_ms = len(pcm_chunk) / (self.sample_rate * 2) * 1000
        is_speech = self._detect_speech(pcm_chunk)

        if is_speech:
            self._silence_ms = 0
            if not self._is_speaking:
                self._speech_ms += chunk_ms
                if self._speech_ms >= self.min_speech_ms:
                    self._is_speaking = True
                    # Include pre-speech padding
                    self._speech_buffer = list(self._pre_buffer)
                    self._speech_buffer.append(pcm_chunk)
                else:
                    self._pre_buffer.append(pcm_chunk)
            else:
                self._speech_buffer.append(pcm_chunk)
        else:
            self._speech_ms = 0
            if self._is_speaking:
                self._silence_ms += chunk_ms
                self._speech_buffer.append(pcm_chunk)
                if self._silence_ms >= self.min_silence_ms:
                    # Speech segment complete
                    segment = b"".join(self._speech_buffer)
                    self._speech_buffer.clear()
                    self._is_speaking = False
                    self._silence_ms = 0
                    return segment
            else:
                self._pre_buffer.append(pcm_chunk)

        return None

    def flush(self) -> Optional[bytes]:
        """Flush any remaining speech buffer."""
        if self._speech_buffer:
            segment = b"".join(self._speech_buffer)
            self._speech_buffer.clear()
            self._is_speaking = False
            return segment
        return None

    def _detect_speech(self, pcm_chunk: bytes) -> bool:
        if self._use_silero:
            return self._detect_silero(pcm_chunk)
        return self._detect_energy(pcm_chunk)

    def _detect_silero(self, pcm_chunk: bytes) -> bool:
        try:
            import torch

            audio = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32767.0
            tensor = torch.from_numpy(audio)
            prob = self._silero_model(tensor, self.sample_rate).item()
            return prob >= self.threshold
        except Exception:
            return self._detect_energy(pcm_chunk)

    def _detect_energy(self, pcm_chunk: bytes) -> bool:
        """Simple energy-based VAD fallback."""
        if len(pcm_chunk) < 2:
            return False
        samples = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples**2)) / 32767.0
        # Energy threshold — much simpler than silero but works for basic use
        return rms > 0.02
