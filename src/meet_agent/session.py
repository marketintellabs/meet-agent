"""Session orchestrator — ties the meeting connector and processing pipeline together."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from meet_agent.audio import rms_level
from meet_agent.config import Settings
from meet_agent.connector.base import MeetingConnector
from meet_agent.pipeline.llm import LLMProvider
from meet_agent.pipeline.stt import STTProvider
from meet_agent.pipeline.tts import TTSProvider
from meet_agent.pipeline.vad import VADProcessor

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    STOPPED = "stopped"


@dataclass
class TranscriptEntry:
    speaker: str
    text: str
    timestamp: float


@dataclass
class SessionInfo:
    id: str
    meeting_url: str
    state: SessionState
    created_at: float
    transcript: list[TranscriptEntry] = field(default_factory=list)
    agent_responses: int = 0
    error: Optional[str] = None


class MeetingSession:
    """Orchestrates a single meeting session.

    State machine:
        IDLE -> LISTENING -> THINKING -> SPEAKING -> LISTENING -> ...
                                                  |-> STOPPED

    Audio flow:
        meeting audio -> VAD -> speech segment -> STT -> LLM -> TTS -> play back
    """

    def __init__(
        self,
        connector: MeetingConnector,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        vad: VADProcessor,
        settings: Settings,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.connector = connector
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.settings = settings

        self.state = SessionState.IDLE
        self.meeting_url = ""
        self.created_at = time.time()
        self.transcript: list[TranscriptEntry] = []
        self.agent_responses = 0
        self.error: Optional[str] = None

        self._stop_event = asyncio.Event()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        self._response_lock = asyncio.Lock()
        self._last_speech_time: float = 0

    @property
    def info(self) -> SessionInfo:
        return SessionInfo(
            id=self.id,
            meeting_url=self.meeting_url,
            state=self.state,
            created_at=self.created_at,
            transcript=list(self.transcript),
            agent_responses=self.agent_responses,
            error=self.error,
        )

    async def start(self, meeting_url: str) -> None:
        """Join the meeting and start the processing loop."""
        self.meeting_url = meeting_url
        self.state = SessionState.LISTENING

        # Wire up audio callback from connector
        self.connector.on_audio(self._on_audio_chunk)
        self.connector.on_event(self._on_meeting_event)

        try:
            await self.connector.join(meeting_url)
            logger.info("Session %s: joined %s", self.id, meeting_url)

            # Run the main processing loop
            await self._processing_loop()
        except Exception as e:
            self.error = str(e)
            logger.exception("Session %s: error", self.id)
            raise
        finally:
            self.state = SessionState.STOPPED
            await self.connector.stop()

    async def stop(self) -> None:
        """Stop the session gracefully."""
        logger.info("Session %s: stopping", self.id)
        self._stop_event.set()

    async def _on_audio_chunk(self, pcm_chunk: bytes) -> None:
        """Called by the connector for each audio chunk from the meeting."""
        try:
            self._audio_queue.put_nowait(pcm_chunk)
        except asyncio.QueueFull:
            pass  # Drop old audio if we can't keep up

    async def _on_meeting_event(self, event: dict[str, Any]) -> None:
        """Handle meeting lifecycle events."""
        event_type = event.get("type", "")
        logger.info("Session %s: meeting event — %s", self.id, event_type)

        if event_type == "disconnected":
            self._stop_event.set()

    async def _processing_loop(self) -> None:
        """Main loop: drain audio queue, run VAD, and trigger responses."""
        while not self._stop_event.is_set():
            try:
                pcm_chunk = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                # Check if we have pending speech and enough silence has passed
                await self._check_response_trigger()
                continue

            if self.state == SessionState.SPEAKING:
                continue  # Don't process audio while we're speaking

            # Run VAD
            segment = self.vad.process_chunk(pcm_chunk)
            if segment:
                self._last_speech_time = time.time()
                await self._handle_speech_segment(segment)

        # Flush any remaining audio
        remaining = self.vad.flush()
        if remaining:
            await self._handle_speech_segment(remaining)

    async def _check_response_trigger(self) -> None:
        """Check if enough silence has passed after speech to trigger a response."""
        if (
            self._last_speech_time > 0
            and self.state == SessionState.LISTENING
            and time.time() - self._last_speech_time
            > self.settings.response_delay_ms / 1000
        ):
            # Silence threshold reached — but only respond if there's new user input
            self._last_speech_time = 0

    async def _handle_speech_segment(self, segment: bytes) -> None:
        """Process a complete speech segment: STT -> LLM -> TTS -> play."""
        if rms_level(segment) < 0.01:
            return  # Skip near-silent segments

        # Don't overlap with an active response
        if self._response_lock.locked():
            return

        async with self._response_lock:
            try:
                # STT
                self.state = SessionState.LISTENING
                text = await self.stt.transcribe(segment, self.settings.sample_rate)
                if not text or len(text.strip()) < 2:
                    return

                logger.info("Session %s: heard — %s", self.id, text[:100])
                self.transcript.append(
                    TranscriptEntry(
                        speaker="Participant", text=text, timestamp=time.time()
                    )
                )

                # LLM
                self.state = SessionState.THINKING
                response = await self.llm.generate_response(text)
                if not response:
                    self.state = SessionState.LISTENING
                    return

                logger.info("Session %s: responding — %s", self.id, response[:100])
                self.transcript.append(
                    TranscriptEntry(
                        speaker=self.settings.agent_name,
                        text=response,
                        timestamp=time.time(),
                    )
                )

                # TTS
                self.state = SessionState.SPEAKING
                audio = await self.tts.synthesize(response)

                # Play into meeting
                await self.connector.play_audio(audio, self.settings.sample_rate)
                self.agent_responses += 1

            except Exception:
                logger.exception("Session %s: pipeline error", self.id)
            finally:
                self.state = SessionState.LISTENING
