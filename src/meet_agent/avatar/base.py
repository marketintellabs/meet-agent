"""Abstract base class for avatar rendering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class AvatarRenderer(ABC):
    """Generates lip-synced video frames from audio + a reference portrait.

    Implementations connect to a GPU backend (local, RunPod, EC2)
    to run real-time avatar inference.
    """

    @abstractmethod
    async def initialize(self, portrait_path: str) -> None:
        """Load the reference portrait and prepare the rendering pipeline.

        Args:
            portrait_path: Path to the reference portrait image (PNG/JPG).
        """
        ...

    @abstractmethod
    async def render_frames(
        self, audio_pcm: bytes, sample_rate: int = 16000
    ) -> list[bytes]:
        """Generate lip-synced video frames for the given audio.

        Args:
            audio_pcm: Raw PCM int16 audio data.
            sample_rate: Audio sample rate.

        Returns:
            List of RGBA frame bytes, each frame at the configured resolution.
        """
        ...

    @abstractmethod
    async def get_idle_frame(self) -> Optional[bytes]:
        """Return a single idle/neutral frame (no lip movement).

        Used when the agent is listening or thinking.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release GPU resources and clean up."""
        ...

    @property
    @abstractmethod
    def frame_width(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_height(self) -> int:
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        ...
