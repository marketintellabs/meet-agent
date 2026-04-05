"""Tests for configuration loading."""

import os

from meet_agent.config import Settings, STTProviderType, TTSProviderType


def test_default_settings():
    settings = Settings(llm_api_key="test-key")
    assert settings.llm_api_key == "test-key"
    assert settings.stt_provider == STTProviderType.DEEPINFRA
    assert settings.tts_provider == TTSProviderType.OPENAI
    assert settings.agent_name == "MeetAgent"
    assert settings.avatar_enabled is False


def test_effective_stt_api_key_fallback():
    settings = Settings(llm_api_key="llm-key", stt_api_key=None)
    assert settings.effective_stt_api_key == "llm-key"


def test_effective_stt_api_key_explicit():
    settings = Settings(llm_api_key="llm-key", stt_api_key="stt-key")
    assert settings.effective_stt_api_key == "stt-key"


def test_effective_stt_api_base_deepinfra():
    settings = Settings(stt_provider=STTProviderType.DEEPINFRA)
    assert "deepinfra" in settings.effective_stt_api_base


def test_effective_stt_api_base_openai():
    settings = Settings(stt_provider=STTProviderType.OPENAI)
    assert "openai" in settings.effective_stt_api_base
