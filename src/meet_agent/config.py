"""Configuration for MeetAgent via environment variables or .env file."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTProviderType(str, Enum):
    FASTER_WHISPER = "faster-whisper"
    DEEPINFRA = "deepinfra"
    OPENAI = "openai"


class TTSProviderType(str, Enum):
    PIPER = "piper"
    OPENAI = "openai"


class GPUProviderType(str, Enum):
    LOCAL = "local"
    RUNPOD = "runpod"
    EC2 = "ec2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_api_base: str = "https://api.deepinfra.com/v1/openai"
    llm_api_key: str = ""
    llm_model: str = "nvidia/Nemotron-Mini-4B-Instruct"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7

    # STT
    stt_provider: STTProviderType = STTProviderType.DEEPINFRA
    stt_api_key: Optional[str] = None
    stt_api_base: Optional[str] = None
    stt_model: str = "openai/whisper-large-v3-turbo"
    stt_language: str = "en"

    # TTS
    tts_provider: TTSProviderType = TTSProviderType.OPENAI
    tts_api_key: Optional[str] = None
    tts_api_base: Optional[str] = None
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_speed: float = 1.0

    # Agent identity
    agent_name: str = "MeetAgent"
    agent_system_prompt: str = (
        "You are a helpful AI meeting assistant. Listen to the conversation "
        "and contribute when addressed or when you have relevant information to share. "
        "Keep responses concise and conversational — you are in a live meeting."
    )

    # VAD
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 700
    vad_pre_speech_pad_ms: int = 300

    # Session
    response_delay_ms: int = 1500
    max_history_turns: int = 50
    sample_rate: int = 16000

    # Avatar (v0.2)
    avatar_enabled: bool = False
    avatar_image: Optional[str] = None
    avatar_gpu_provider: GPUProviderType = GPUProviderType.LOCAL

    # RunPod (v0.2)
    runpod_api_key: Optional[str] = None
    runpod_endpoint_id: Optional[str] = None

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = Field(default=8080)

    @property
    def effective_stt_api_key(self) -> str:
        return self.stt_api_key or self.llm_api_key

    @property
    def effective_stt_api_base(self) -> str:
        if self.stt_api_base:
            return self.stt_api_base
        if self.stt_provider == STTProviderType.DEEPINFRA:
            return "https://api.deepinfra.com/v1"
        if self.stt_provider == STTProviderType.OPENAI:
            return "https://api.openai.com/v1"
        return ""

    @property
    def effective_tts_api_key(self) -> str:
        return self.tts_api_key or self.llm_api_key

    @property
    def effective_tts_api_base(self) -> str:
        if self.tts_api_base:
            return self.tts_api_base
        if self.tts_provider == TTSProviderType.OPENAI:
            return "https://api.openai.com/v1"
        return ""


def get_settings() -> Settings:
    return Settings()
