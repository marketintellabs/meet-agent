"""AWS EC2 GPU provider — spin up GPU instances on demand."""

from __future__ import annotations

import logging
from typing import Optional

from meet_agent.gpu.base import GPUProvider

logger = logging.getLogger(__name__)


class EC2GPUProvider(GPUProvider):
    """Manages an EC2 GPU instance (g6.xlarge) for avatar rendering.

    v0.2 stub — full implementation will use boto3 to:
    1. Launch a g6.xlarge spot instance with the LiveTalking AMI
    2. Wait for the instance to be ready
    3. Return the instance's public IP as the server URL
    4. Terminate the instance on stop()

    For now, this expects a pre-running EC2 instance.
    """

    def __init__(self, instance_url: str = "") -> None:
        self.instance_url = instance_url
        self._running = False

    async def start(self) -> str:
        if not self.instance_url:
            raise RuntimeError(
                "EC2 GPU provider requires a pre-running instance URL. "
                "Set AVATAR_EC2_URL or launch a g6.xlarge with LiveTalking manually."
            )
        logger.info("Using EC2 GPU at %s", self.instance_url)
        self._running = True
        return self.instance_url

    async def stop(self) -> None:
        self._running = False

    async def health_check(self) -> bool:
        if not self.instance_url:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.instance_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        return self._running
