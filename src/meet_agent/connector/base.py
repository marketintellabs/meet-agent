"""Abstract base class for meeting platform connectors."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

AudioCallback = Callable[[bytes], Coroutine[Any, Any, None]]
EventCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class ConnectorState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    WAITING = "waiting"  # waiting room / ask-to-join
    CONNECTED = "connected"
    LEAVING = "leaving"


class MeetingConnector(ABC):
    """Abstract interface for joining meetings and streaming audio."""

    def __init__(self, agent_name: str = "MeetAgent") -> None:
        self.agent_name = agent_name
        self.state = ConnectorState.DISCONNECTED
        self._audio_callbacks: list[AudioCallback] = []
        self._event_callbacks: list[EventCallback] = []
        self._stop_event = asyncio.Event()

    def on_audio(self, callback: AudioCallback) -> None:
        """Register a callback for incoming audio chunks (int16 PCM, 16kHz mono)."""
        self._audio_callbacks.append(callback)

    def on_event(self, callback: EventCallback) -> None:
        """Register a callback for meeting events (join, leave, chat, etc.)."""
        self._event_callbacks.append(callback)

    async def _emit_audio(self, pcm_chunk: bytes) -> None:
        for cb in self._audio_callbacks:
            try:
                await cb(pcm_chunk)
            except Exception:
                logger.exception("Error in audio callback")

    async def _emit_event(self, event: dict[str, Any]) -> None:
        for cb in self._event_callbacks:
            try:
                await cb(event)
            except Exception:
                logger.exception("Error in event callback")

    @abstractmethod
    async def join(self, meeting_url: str) -> None:
        """Join a meeting. Blocks until connected or raises on failure."""
        ...

    @abstractmethod
    async def leave(self) -> None:
        """Leave the current meeting and clean up resources."""
        ...

    @abstractmethod
    async def play_audio(self, pcm_data: bytes, sample_rate: int = 16000) -> None:
        """Play audio into the meeting (int16 PCM)."""
        ...

    @abstractmethod
    async def set_video_frame(self, frame_rgba: bytes, width: int, height: int) -> None:
        """Set the current video frame for the agent's camera feed (v0.2)."""
        ...

    async def stop(self) -> None:
        """Signal the connector to stop."""
        self._stop_event.set()
        if self.state in (ConnectorState.CONNECTED, ConnectorState.WAITING):
            await self.leave()

    @staticmethod
    def detect_platform(url: str) -> str:
        """Detect meeting platform from URL."""
        if "meet.google.com" in url:
            return "google_meet"
        if "zoom.us" in url:
            return "zoom"
        if "teams.microsoft.com" in url:
            return "teams"
        raise ValueError(f"Unsupported meeting URL: {url}")
