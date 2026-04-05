"""CLI entry point for MeetAgent."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from meet_agent import __version__
from meet_agent.config import Settings, get_settings

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def _create_session_from_settings(settings: Settings):
    """Wire up all components and return a MeetingSession."""
    from meet_agent.connector.base import MeetingConnector
    from meet_agent.connector.google_meet import GoogleMeetConnector
    from meet_agent.pipeline.llm import LLMProvider
    from meet_agent.pipeline.stt import create_stt_provider
    from meet_agent.pipeline.tts import create_tts_provider
    from meet_agent.pipeline.vad import VADProcessor
    from meet_agent.session import MeetingSession

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

    connector: MeetingConnector = GoogleMeetConnector(agent_name=settings.agent_name)

    return MeetingSession(
        connector=connector,
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        settings=settings,
    )


@click.group()
@click.version_option(version=__version__, prog_name="meet-agent")
def main() -> None:
    """MeetAgent — Open-source AI agents for video meetings."""
    pass


@main.command()
@click.argument("meeting_url")
@click.option("--name", default=None, help="Agent display name in the meeting")
@click.option("--system-prompt", default=None, help="System prompt for the LLM")
@click.option("--headless/--no-headless", default=True, help="Run browser in headless mode")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def join(
    meeting_url: str,
    name: str | None,
    system_prompt: str | None,
    headless: bool,
    verbose: bool,
) -> None:
    """Join a meeting and participate as an AI agent.

    MEETING_URL is a Google Meet or Zoom meeting link.

    Examples:

        meet-agent join "https://meet.google.com/abc-defg-hij"

        meet-agent join "https://zoom.us/j/123456789" --name "Financial Analyst"
    """
    _setup_logging(verbose)
    settings = get_settings()

    if name:
        settings.agent_name = name
    if system_prompt:
        settings.agent_system_prompt = system_prompt

    console.print(
        Panel(
            f"[bold]MeetAgent v{__version__}[/bold]\n\n"
            f"Meeting:  {meeting_url}\n"
            f"Agent:    {settings.agent_name}\n"
            f"LLM:      {settings.llm_model}\n"
            f"STT:      {settings.stt_provider.value}\n"
            f"TTS:      {settings.tts_provider.value}",
            title="Joining Meeting",
            border_style="blue",
        )
    )

    # Detect platform and create the right connector
    from meet_agent.connector.base import MeetingConnector
    from meet_agent.connector.google_meet import GoogleMeetConnector
    from meet_agent.connector.zoom import ZoomConnector
    from meet_agent.pipeline.llm import LLMProvider
    from meet_agent.pipeline.stt import create_stt_provider
    from meet_agent.pipeline.tts import create_tts_provider
    from meet_agent.pipeline.vad import VADProcessor
    from meet_agent.session import MeetingSession

    platform = MeetingConnector.detect_platform(meeting_url)

    connector: MeetingConnector
    if platform == "google_meet":
        connector = GoogleMeetConnector(agent_name=settings.agent_name, headless=headless)
    elif platform == "zoom":
        connector = ZoomConnector(agent_name=settings.agent_name, headless=headless)
    else:
        console.print(f"[red]Unsupported platform: {platform}[/red]")
        sys.exit(1)

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

    async def _run() -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(session.stop()))
        await session.start(meeting_url)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Session ended.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)

    # Print summary
    info = session.info
    console.print(
        Panel(
            f"Duration:       {info.created_at:.0f}s\n"
            f"Transcript:     {len(info.transcript)} entries\n"
            f"Agent responses: {info.agent_responses}",
            title="Session Summary",
            border_style="green",
        )
    )


@main.command()
@click.option("--host", default="0.0.0.0", help="Server bind host")
@click.option("--port", default=8080, type=int, help="Server bind port")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def serve(host: str, port: int, verbose: bool) -> None:
    """Start the MeetAgent API server.

    Provides a REST API for creating and managing meeting sessions programmatically.
    """
    _setup_logging(verbose)
    import uvicorn

    from meet_agent.server import app

    console.print(
        Panel(
            f"[bold]MeetAgent API Server[/bold]\n\n"
            f"Listening on http://{host}:{port}\n"
            f"Docs at http://{host}:{port}/docs",
            title="Server",
            border_style="blue",
        )
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
