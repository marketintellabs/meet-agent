"""Basic example: join a Google Meet as a helpful assistant.

Usage:
    export LLM_API_KEY=your-deepinfra-key
    python examples/basic_assistant.py "https://meet.google.com/abc-defg-hij"
"""

import asyncio
import sys

from meet_agent.config import get_settings
from meet_agent.connector.base import MeetingConnector
from meet_agent.connector.google_meet import GoogleMeetConnector
from meet_agent.connector.zoom import ZoomConnector
from meet_agent.pipeline.llm import LLMProvider
from meet_agent.pipeline.stt import create_stt_provider
from meet_agent.pipeline.tts import create_tts_provider
from meet_agent.pipeline.vad import VADProcessor
from meet_agent.session import MeetingSession


async def main():
    if len(sys.argv) < 2:
        print("Usage: python basic_assistant.py <meeting-url>")
        sys.exit(1)

    meeting_url = sys.argv[1]
    settings = get_settings()

    platform = MeetingConnector.detect_platform(meeting_url)
    if platform == "google_meet":
        connector = GoogleMeetConnector(agent_name="Assistant", headless=False)
    elif platform == "zoom":
        connector = ZoomConnector(agent_name="Assistant", headless=False)
    else:
        print(f"Unsupported: {platform}")
        sys.exit(1)

    stt = create_stt_provider(
        provider=settings.stt_provider.value,
        api_key=settings.effective_stt_api_key,
        api_base=settings.effective_stt_api_base,
    )
    llm = LLMProvider(
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base,
        model=settings.llm_model,
        system_prompt="You are a helpful meeting assistant. Be concise.",
    )
    tts = create_tts_provider(
        provider=settings.tts_provider.value,
        api_key=settings.effective_tts_api_key,
    )
    vad = VADProcessor()

    session = MeetingSession(
        connector=connector, stt=stt, llm=llm, tts=tts, vad=vad, settings=settings
    )

    try:
        await session.start(meeting_url)
    except KeyboardInterrupt:
        await session.stop()
        print(f"\nTranscript: {len(session.transcript)} entries")
        print(f"Agent spoke: {session.agent_responses} times")


if __name__ == "__main__":
    asyncio.run(main())
