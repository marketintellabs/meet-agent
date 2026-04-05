"""LiveTalking avatar renderer — real-time lip-sync via LiveTalking backend.

Connects to a running LiveTalking instance (local or remote GPU)
and streams audio to get back lip-synced video frames.

Requires a GPU backend running LiveTalking. See:
https://github.com/lipku/LiveTalking

This is a v0.2 feature — the interface is implemented but requires
a running LiveTalking server to function.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from meet_agent.avatar.base import AvatarRenderer

logger = logging.getLogger(__name__)


class LiveTalkingRenderer(AvatarRenderer):
    """Real-time lip-sync avatar using a LiveTalking backend server.

    The LiveTalking server must be running and accessible at the configured URL.
    It handles the GPU-intensive wav2lip/MuseTalk inference.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8010",
        width: int = 640,
        height: int = 480,
        model: str = "wav2lip",
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self._width = width
        self._height = height
        self.model = model
        self._client = httpx.AsyncClient(timeout=30.0)
        self._session_id: Optional[str] = None
        self._idle_frame: Optional[bytes] = None

    async def initialize(self, portrait_path: str) -> None:
        """Upload the reference portrait to the LiveTalking server."""
        with open(portrait_path, "rb") as f:
            portrait_data = f.read()

        resp = await self._client.post(
            f"{self.server_url}/api/init",
            files={"portrait": ("portrait.png", portrait_data, "image/png")},
            data={"model": self.model, "width": self._width, "height": self._height},
        )
        resp.raise_for_status()
        result = resp.json()
        self._session_id = result.get("session_id")
        logger.info("LiveTalking session initialized: %s", self._session_id)

        # Get the idle frame
        idle_resp = await self._client.get(
            f"{self.server_url}/api/idle_frame",
            params={"session_id": self._session_id},
        )
        if idle_resp.status_code == 200:
            self._idle_frame = idle_resp.content

    async def render_frames(
        self, audio_pcm: bytes, sample_rate: int = 16000
    ) -> list[bytes]:
        """Send audio to LiveTalking and receive lip-synced frames."""
        if not self._session_id:
            raise RuntimeError("Avatar not initialized — call initialize() first")

        resp = await self._client.post(
            f"{self.server_url}/api/render",
            data={
                "session_id": self._session_id,
                "sample_rate": sample_rate,
            },
            files={"audio": ("audio.pcm", audio_pcm, "audio/pcm")},
            timeout=60.0,
        )
        resp.raise_for_status()
        result = resp.json()

        frames: list[bytes] = []
        for frame_b64 in result.get("frames", []):
            import base64
            frames.append(base64.b64decode(frame_b64))
        return frames

    async def get_idle_frame(self) -> Optional[bytes]:
        return self._idle_frame

    async def shutdown(self) -> None:
        if self._session_id:
            try:
                await self._client.post(
                    f"{self.server_url}/api/destroy",
                    json={"session_id": self._session_id},
                )
            except Exception:
                pass
        await self._client.aclose()

    @property
    def frame_width(self) -> int:
        return self._width

    @property
    def frame_height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return 25.0
