"""Speech-to-Text provider interface and implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

from meet_agent.audio import pcm_to_wav

logger = logging.getLogger(__name__)


class STTProvider(ABC):
    """Abstract interface for speech-to-text transcription."""

    @abstractmethod
    async def transcribe(self, pcm_audio: bytes, sample_rate: int = 16000) -> str:
        """Transcribe PCM int16 audio to text."""
        ...


class DeepInfraSTT(STTProvider):
    """STT via DeepInfra's Whisper API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.deepinfra.com/v1",
        model: str = "openai/whisper-large-v3-turbo",
        language: str = "en",
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.language = language
        self._client = httpx.AsyncClient(timeout=30.0)

    async def transcribe(self, pcm_audio: bytes, sample_rate: int = 16000) -> str:
        wav_data = pcm_to_wav(pcm_audio, sample_rate)
        url = f"{self.api_base}/audio/transcriptions"

        resp = await self._client.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data={"model": self.model, "language": self.language},
            files={"file": ("audio.wav", wav_data, "audio/wav")},
        )
        resp.raise_for_status()
        result = resp.json()
        text = result.get("text", "").strip()
        logger.debug("STT [DeepInfra]: %s", text[:100])
        return text


class OpenAISTT(STTProvider):
    """STT via OpenAI's Whisper API."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "whisper-1",
        language: str = "en",
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.language = language
        self._client = httpx.AsyncClient(timeout=30.0)

    async def transcribe(self, pcm_audio: bytes, sample_rate: int = 16000) -> str:
        wav_data = pcm_to_wav(pcm_audio, sample_rate)
        url = f"{self.api_base}/audio/transcriptions"

        resp = await self._client.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data={"model": self.model, "language": self.language},
            files={"file": ("audio.wav", wav_data, "audio/wav")},
        )
        resp.raise_for_status()
        result = resp.json()
        text = result.get("text", "").strip()
        logger.debug("STT [OpenAI]: %s", text[:100])
        return text


class FasterWhisperSTT(STTProvider):
    """STT via locally-running faster-whisper. Requires `pip install meet-agent[stt-local]`."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ) -> None:
        self.language = language
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. Install with: pip install meet-agent[stt-local]"
            ) from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    async def transcribe(self, pcm_audio: bytes, sample_rate: int = 16000) -> str:
        import asyncio

        import numpy as np

        audio = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32767.0

        def _run() -> str:
            segments, _ = self._model.transcribe(audio, language=self.language)
            return " ".join(seg.text.strip() for seg in segments).strip()

        text = await asyncio.get_event_loop().run_in_executor(None, _run)
        logger.debug("STT [faster-whisper]: %s", text[:100])
        return text


def create_stt_provider(
    provider: str,
    api_key: str = "",
    api_base: str = "",
    model: str = "",
    language: str = "en",
) -> STTProvider:
    """Factory to create an STT provider by name."""
    if provider == "deepinfra":
        return DeepInfraSTT(
            api_key=api_key,
            api_base=api_base or "https://api.deepinfra.com/v1",
            model=model or "openai/whisper-large-v3-turbo",
            language=language,
        )
    if provider == "openai":
        return OpenAISTT(
            api_key=api_key,
            api_base=api_base or "https://api.openai.com/v1",
            model=model or "whisper-1",
            language=language,
        )
    if provider == "faster-whisper":
        return FasterWhisperSTT(model_size=model or "base.en", language=language)
    raise ValueError(f"Unknown STT provider: {provider}")
