"""Static avatar renderer — displays a fixed portrait image (no lip-sync).

This is the zero-GPU fallback: the agent shows a static profile picture
in the meeting while speaking. No GPU or ML inference required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from meet_agent.avatar.base import AvatarRenderer

logger = logging.getLogger(__name__)


class StaticAvatarRenderer(AvatarRenderer):
    """Displays a static portrait image. No lip-sync, no GPU needed."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width = width
        self._height = height
        self._portrait_data: Optional[bytes] = None

    async def initialize(self, portrait_path: str) -> None:
        path = Path(portrait_path)
        if not path.exists():
            raise FileNotFoundError(f"Portrait not found: {portrait_path}")
        self._portrait_data = path.read_bytes()
        logger.info("Static avatar loaded: %s (%d bytes)", portrait_path, len(self._portrait_data))

    async def render_frames(
        self, audio_pcm: bytes, sample_rate: int = 16000
    ) -> list[bytes]:
        # Static renderer always returns the same frame
        if self._portrait_data:
            duration_s = len(audio_pcm) / (sample_rate * 2)
            n_frames = max(1, int(duration_s * self.fps))
            return [self._portrait_data] * n_frames
        return []

    async def get_idle_frame(self) -> Optional[bytes]:
        return self._portrait_data

    async def shutdown(self) -> None:
        self._portrait_data = None

    @property
    def frame_width(self) -> int:
        return self._width

    @property
    def frame_height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return 25.0
