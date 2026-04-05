"""Local GPU provider — for development with a local NVIDIA GPU."""

from __future__ import annotations

import logging

import httpx

from meet_agent.gpu.base import GPUProvider

logger = logging.getLogger(__name__)


class LocalGPUProvider(GPUProvider):
    """Assumes a locally-running LiveTalking server on the same machine.

    For development: start LiveTalking manually, then point MeetAgent at it.
    """

    def __init__(self, server_url: str = "http://localhost:8010") -> None:
        self.server_url = server_url
        self._running = False

    async def start(self) -> str:
        logger.info("Using local GPU at %s", self.server_url)
        if await self.health_check():
            self._running = True
            return self.server_url
        raise RuntimeError(
            f"Local GPU server not reachable at {self.server_url}. "
            "Start LiveTalking manually: python app.py --model wav2lip"
        )

    async def stop(self) -> None:
        self._running = False

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.server_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        return self._running
