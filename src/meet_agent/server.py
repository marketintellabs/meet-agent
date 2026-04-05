"""FastAPI server for managing meeting sessions programmatically."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from meet_agent import __version__
from meet_agent.config import Settings, get_settings
from meet_agent.connector.base import MeetingConnector
from meet_agent.connector.google_meet import GoogleMeetConnector
from meet_agent.connector.zoom import ZoomConnector
from meet_agent.pipeline.llm import LLMProvider
from meet_agent.pipeline.stt import create_stt_provider
from meet_agent.pipeline.tts import create_tts_provider
from meet_agent.pipeline.vad import VADProcessor
from meet_agent.session import MeetingSession, SessionState

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MeetAgent API",
    version=__version__,
    description="REST API for managing AI meeting agents",
)

# In-memory session store
_sessions: dict[str, MeetingSession] = {}
_session_tasks: dict[str, asyncio.Task] = {}


class JoinRequest(BaseModel):
    meeting_url: str = Field(..., description="Google Meet or Zoom meeting URL")
    agent_name: Optional[str] = Field(None, description="Display name for the agent")
    system_prompt: Optional[str] = Field(None, description="System prompt for the LLM")


class SessionResponse(BaseModel):
    id: str
    meeting_url: str
    state: str
    created_at: float
    agent_responses: int
    error: Optional[str] = None


class TranscriptEntryResponse(BaseModel):
    speaker: str
    text: str
    timestamp: float


class TranscriptResponse(BaseModel):
    session_id: str
    entries: list[TranscriptEntryResponse]


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__}


@app.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(req: JoinRequest):
    """Create a new meeting session. The agent will join the meeting immediately."""
    settings = get_settings()
    if req.agent_name:
        settings.agent_name = req.agent_name
    if req.system_prompt:
        settings.agent_system_prompt = req.system_prompt

    platform = MeetingConnector.detect_platform(req.meeting_url)
    connector: MeetingConnector
    if platform == "google_meet":
        connector = GoogleMeetConnector(agent_name=settings.agent_name)
    elif platform == "zoom":
        connector = ZoomConnector(agent_name=settings.agent_name)
    else:
        raise HTTPException(400, f"Unsupported meeting platform: {platform}")

    stt = create_stt_provider(
        provider=settings.stt_provider.value,
        api_key=settings.effective_stt_api_key,
        api_base=settings.effective_stt_api_base,
        model=settings.stt_model,
        language=settings.stt_language,
    )
    llm = LLMProvider(
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        system_prompt=settings.agent_system_prompt,
        max_history_turns=settings.max_history_turns,
    )
    tts = create_tts_provider(
        provider=settings.tts_provider.value,
        api_key=settings.effective_tts_api_key,
        api_base=settings.effective_tts_api_base,
        model=settings.tts_model,
        voice=settings.tts_voice,
        speed=settings.tts_speed,
    )
    vad = VADProcessor(
        threshold=settings.vad_threshold,
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_ms=settings.vad_min_silence_ms,
        pre_speech_pad_ms=settings.vad_pre_speech_pad_ms,
        sample_rate=settings.sample_rate,
    )

    session = MeetingSession(
        connector=connector, stt=stt, llm=llm, tts=tts, vad=vad, settings=settings
    )
    _sessions[session.id] = session

    async def _run():
        try:
            await session.start(req.meeting_url)
        except Exception:
            logger.exception("Session %s failed", session.id)
        finally:
            _session_tasks.pop(session.id, None)

    task = asyncio.create_task(_run())
    _session_tasks[session.id] = task

    # Wait briefly for connection to establish
    await asyncio.sleep(2)

    info = session.info
    return SessionResponse(
        id=info.id,
        meeting_url=info.meeting_url,
        state=info.state.value,
        created_at=info.created_at,
        agent_responses=info.agent_responses,
        error=info.error,
    )


@app.get("/sessions", response_model=list[SessionResponse])
async def list_sessions():
    """List all active and recent sessions."""
    results = []
    for session in _sessions.values():
        info = session.info
        results.append(
            SessionResponse(
                id=info.id,
                meeting_url=info.meeting_url,
                state=info.state.value,
                created_at=info.created_at,
                agent_responses=info.agent_responses,
                error=info.error,
            )
        )
    return results


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get the status of a specific session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    info = session.info
    return SessionResponse(
        id=info.id,
        meeting_url=info.meeting_url,
        state=info.state.value,
        created_at=info.created_at,
        agent_responses=info.agent_responses,
        error=info.error,
    )


@app.get("/sessions/{session_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(session_id: str):
    """Get the full transcript for a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return TranscriptResponse(
        session_id=session_id,
        entries=[
            TranscriptEntryResponse(
                speaker=e.speaker, text=e.text, timestamp=e.timestamp
            )
            for e in session.transcript
        ],
    )


@app.post("/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    """Stop a running session and leave the meeting."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    await session.stop()
    return {"status": "stopping", "session_id": session_id}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Stop and remove a session."""
    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(404, "Session not found")
    await session.stop()
    task = _session_tasks.pop(session_id, None)
    if task:
        task.cancel()
    return {"status": "deleted", "session_id": session_id}
