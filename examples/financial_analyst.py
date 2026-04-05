"""MarketIntelLabs showcase: Financial Analyst meeting agent.

A specialized agent that joins meetings with deep financial knowledge,
can discuss market trends, analyze investment strategies, and provide
real-time financial insights during team meetings.

Usage:
    export LLM_API_KEY=your-deepinfra-key
    export TTS_API_KEY=your-openai-key
    python examples/financial_analyst.py "https://meet.google.com/abc-defg-hij"
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

FINANCIAL_ANALYST_PROMPT = """\
You are a senior financial analyst AI assistant from MarketIntelLabs, \
participating in a live meeting. Your expertise includes:

- Precious metals markets (gold, silver, platinum)
- Equity analysis and market trends
- Macroeconomic indicators and their market impact
- Technical and fundamental analysis
- Risk assessment and portfolio strategy

Guidelines for the meeting:
- Keep responses concise (2-3 sentences max) — this is a live conversation
- Reference specific data points, percentages, and timeframes when relevant
- Flag risks and uncertainties clearly
- If asked about something outside your expertise, say so
- Use professional but accessible language
- When discussing market movements, provide context (e.g., "Gold is up 2.3% \
this week, continuing the trend driven by weakening dollar sentiment")

You represent MarketIntelLabs' commitment to data-driven financial intelligence.\
"""


async def main():
    if len(sys.argv) < 2:
        print("Usage: python financial_analyst.py <meeting-url>")
        sys.exit(1)

    meeting_url = sys.argv[1]
    settings = get_settings()

    platform = MeetingConnector.detect_platform(meeting_url)
    if platform == "google_meet":
        connector = GoogleMeetConnector(
            agent_name="MarketIntelLabs Analyst", headless=False
        )
    elif platform == "zoom":
        connector = ZoomConnector(
            agent_name="MarketIntelLabs Analyst", headless=False
        )
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
        system_prompt=FINANCIAL_ANALYST_PROMPT,
        max_tokens=256,
        temperature=0.5,
    )
    tts = create_tts_provider(
        provider=settings.tts_provider.value,
        api_key=settings.effective_tts_api_key,
    )
    vad = VADProcessor()

    session = MeetingSession(
        connector=connector, stt=stt, llm=llm, tts=tts, vad=vad, settings=settings
    )

    print(f"Joining as MarketIntelLabs Financial Analyst...")
    try:
        await session.start(meeting_url)
    except KeyboardInterrupt:
        await session.stop()
        print(f"\nSession complete.")
        print(f"  Transcript entries: {len(session.transcript)}")
        print(f"  Agent responses:    {session.agent_responses}")


if __name__ == "__main__":
    asyncio.run(main())
