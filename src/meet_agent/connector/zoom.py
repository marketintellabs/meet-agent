"""Zoom connector using Playwright headless browser (web client)."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from meet_agent.connector.base import ConnectorState, MeetingConnector

logger = logging.getLogger(__name__)

# Reuse the same audio capture/playback JS — browser APIs are identical
AUDIO_CAPTURE_JS = """
() => {
    if (window.__meetAgentCapture) return;
    window.__meetAgentCapture = true;

    const ctx = new AudioContext({ sampleRate: 16000 });
    const dest = ctx.createMediaStreamDestination();

    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.tagName === 'AUDIO' || node.tagName === 'VIDEO') {
                    try {
                        const src = ctx.createMediaElementSource(node);
                        src.connect(dest);
                        src.connect(ctx.destination);
                    } catch (e) {}
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    document.querySelectorAll('audio, video').forEach(el => {
        try {
            const src = ctx.createMediaElementSource(el);
            src.connect(dest);
            src.connect(ctx.destination);
        } catch (e) {}
    });

    const processor = ctx.createScriptProcessor(4096, 1, 1);
    dest.stream.getAudioTracks().forEach(track => {
        const src = ctx.createMediaStreamSource(new MediaStream([track]));
        src.connect(processor);
    });
    processor.connect(ctx.destination);

    window.__meetAgentChunks = [];
    processor.onaudioprocess = (e) => {
        const data = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(data.length);
        for (let i = 0; i < data.length; i++) {
            int16[i] = Math.max(-32768, Math.min(32767, Math.round(data[i] * 32767)));
        }
        const b64 = btoa(String.fromCharCode(...new Uint8Array(int16.buffer)));
        window.__meetAgentChunks.push(b64);
    };
}
"""

AUDIO_PLAYBACK_JS = """
(b64Data) => {
    const raw = atob(b64Data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32767;

    const ctx = new AudioContext({ sampleRate: 16000 });
    const buffer = ctx.createBuffer(1, float32.length, 16000);
    buffer.copyToChannel(float32, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start();

    return new Promise(resolve => {
        source.onended = resolve;
        setTimeout(resolve, (float32.length / 16000) * 1000 + 500);
    });
}
"""

DRAIN_CHUNKS_JS = """
() => {
    const chunks = window.__meetAgentChunks || [];
    window.__meetAgentChunks = [];
    return chunks;
}
"""


class ZoomConnector(MeetingConnector):
    """Joins Zoom meetings via the web client using headless Chromium."""

    def __init__(self, agent_name: str = "MeetAgent", headless: bool = True) -> None:
        super().__init__(agent_name)
        self.headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._capture_task: Optional[asyncio.Task] = None

    async def join(self, meeting_url: str) -> None:
        self.state = ConnectorState.CONNECTING
        logger.info("Joining Zoom meeting: %s", meeting_url)

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self._context = await self._browser.new_context(
            permissions=["microphone", "camera"],
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()

        # Zoom links redirect to a launcher page — force web client
        web_url = self._to_web_client_url(meeting_url)
        await self._page.goto(web_url, wait_until="networkidle", timeout=30000)

        # Handle "Join from Your Browser" link if the page shows app download prompt
        await self._force_web_client()

        # Enter name
        await self._set_name()

        # Click join
        await self._click_join()

        self.state = ConnectorState.WAITING
        await self._emit_event({"type": "waiting", "url": meeting_url})

        # Wait for meeting to load
        await self._wait_for_meeting()

        self.state = ConnectorState.CONNECTED
        await self._emit_event({"type": "connected", "url": meeting_url})
        logger.info("Connected to Zoom meeting")

        # Mute camera/mic after joining
        await self._mute_media()

        # Inject audio capture
        await self._page.evaluate(AUDIO_CAPTURE_JS)

        # Start audio polling
        self._capture_task = asyncio.create_task(self._poll_audio())

    async def leave(self) -> None:
        logger.info("Leaving Zoom meeting")
        self.state = ConnectorState.LEAVING
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass

        if self._page:
            try:
                leave_btn = self._page.locator('button:has-text("Leave")').first
                if await leave_btn.is_visible(timeout=2000):
                    await leave_btn.click()
                    # Confirm leave
                    confirm = self._page.locator('button:has-text("Leave Meeting")').first
                    if await confirm.is_visible(timeout=2000):
                        await confirm.click()
            except Exception:
                pass

        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        self.state = ConnectorState.DISCONNECTED
        await self._emit_event({"type": "disconnected"})

    async def play_audio(self, pcm_data: bytes, sample_rate: int = 16000) -> None:
        if not self._page or self.state != ConnectorState.CONNECTED:
            return
        b64 = base64.b64encode(pcm_data).decode("ascii")
        await self._page.evaluate(AUDIO_PLAYBACK_JS, b64)

    async def set_video_frame(self, frame_rgba: bytes, width: int, height: int) -> None:
        pass

    # -- Private helpers --

    @staticmethod
    def _to_web_client_url(url: str) -> str:
        """Convert a Zoom invite URL to the web client URL."""
        # https://us05web.zoom.us/j/12345?pwd=abc -> force web client
        if "/wc/" not in url:
            url = url.replace("/j/", "/wc/join/")
        return url

    async def _force_web_client(self) -> None:
        """Click 'Join from Your Browser' if the app download page is shown."""
        for text in [
            "Join from Your Browser",
            "join from your browser",
            "Launch Meeting",
        ]:
            try:
                link = self._page.locator(f'a:has-text("{text}")').first
                if await link.is_visible(timeout=5000):
                    await link.click()
                    await self._page.wait_for_load_state("networkidle", timeout=15000)
                    return
            except Exception:
                continue

    async def _set_name(self) -> None:
        """Enter the agent's display name."""
        try:
            name_input = self._page.locator("#inputname, input[placeholder*='name' i]").first
            if await name_input.is_visible(timeout=3000):
                await name_input.fill(self.agent_name)
        except Exception:
            logger.debug("Name input not found on Zoom page")

    async def _click_join(self) -> None:
        """Click the join button."""
        for selector in [
            'button:has-text("Join")',
            "#joinBtn",
            'button[type="button"]:has-text("Join")',
        ]:
            try:
                btn = self._page.locator(selector).first
                if await btn.is_visible(timeout=3000):
                    await btn.click()
                    return
            except Exception:
                continue
        raise RuntimeError("Could not find a join button on the Zoom page")

    async def _wait_for_meeting(self) -> None:
        """Wait until the meeting UI is loaded."""
        try:
            await self._page.wait_for_selector(
                "#wc-footer, .meeting-app, .zm-btn--leave",
                timeout=60000,
            )
        except Exception:
            raise RuntimeError(
                "Timed out waiting to join Zoom meeting. "
                "The host may need to admit the bot from the waiting room."
            )

    async def _mute_media(self) -> None:
        """Mute camera and microphone after joining."""
        for label in ["Stop Video", "Mute", "stop my video", "mute my audio"]:
            try:
                btn = self._page.locator(f'button[aria-label*="{label}" i]').first
                if await btn.is_visible(timeout=2000):
                    await btn.click()
            except Exception:
                pass

    async def _poll_audio(self) -> None:
        """Background task: drain captured audio chunks and emit them."""
        while not self._stop_event.is_set():
            try:
                if self._page:
                    chunks = await self._page.evaluate(DRAIN_CHUNKS_JS)
                    for b64_chunk in chunks:
                        raw = base64.b64decode(b64_chunk)
                        await self._emit_audio(raw)
            except Exception:
                logger.debug("Audio poll error", exc_info=True)
            await asyncio.sleep(0.1)
