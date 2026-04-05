"""Text-to-Speech provider interface and implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

from meet_agent.audio import wav_to_pcm

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Abstract interface for text-to-speech synthesis."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to PCM int16 audio at 16kHz mono. Returns raw PCM bytes."""
        ...


class OpenAITTS(TTSProvider):
    """TTS via OpenAI's TTS API (or any compatible endpoint)."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "tts-1",
        voice: str = "alloy",
        speed: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.voice = voice
        self.speed = speed
        self._client = httpx.AsyncClient(timeout=30.0)

    async def synthesize(self, text: str) -> bytes:
        url = f"{self.api_base}/audio/speech"
        resp = await self._client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "speed": self.speed,
                "response_format": "wav",
            },
        )
        resp.raise_for_status()
        wav_data = resp.content
        pcm, sr, ch = wav_to_pcm(wav_data)
        if sr != 16000:
            from meet_agent.audio import float32_to_int16, int16_to_float32, resample

            audio = int16_to_float32(pcm)
            audio = resample(audio, sr, 16000)
            pcm = float32_to_int16(audio)
        logger.debug("TTS [OpenAI]: synthesized %d bytes", len(pcm))
        return pcm


class PiperTTS(TTSProvider):
    """TTS via locally-running Piper. Requires `pip install meet-agent[tts-local]`."""

    def __init__(self, model: str = "en_US-lessac-medium", data_dir: str | None = None) -> None:
        self.model = model
        self.data_dir = data_dir
        self._piper = None

    async def synthesize(self, text: str) -> bytes:
        import asyncio

        def _run() -> bytes:
            import shutil
            import subprocess

            piper_bin = shutil.which("piper") or "piper"
            args = [piper_bin, "--model", self.model, "--output_raw"]
            if self.data_dir:
                args.extend(["--data-dir", self.data_dir])

            proc = subprocess.run(
                args,
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Piper TTS failed: {proc.stderr.decode()}")
            return proc.stdout

        pcm = await asyncio.get_event_loop().run_in_executor(None, _run)
        logger.debug("TTS [Piper]: synthesized %d bytes", len(pcm))
        return pcm


def create_tts_provider(
    provider: str,
    api_key: str = "",
    api_base: str = "",
    model: str = "",
    voice: str = "alloy",
    speed: float = 1.0,
) -> TTSProvider:
    """Factory to create a TTS provider by name."""
    if provider == "openai":
        return OpenAITTS(
            api_key=api_key,
            api_base=api_base or "https://api.openai.com/v1",
            model=model or "tts-1",
            voice=voice,
            speed=speed,
        )
    if provider == "piper":
        return PiperTTS(model=model or "en_US-lessac-medium")
    raise ValueError(f"Unknown TTS provider: {provider}")
