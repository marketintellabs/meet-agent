"""RunPod Serverless GPU provider — pay-per-second GPU inference."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from meet_agent.gpu.base import GPUProvider

logger = logging.getLogger(__name__)


class RunPodGPUProvider(GPUProvider):
    """Runs avatar rendering on RunPod Serverless.

    Deploys a LiveTalking worker as a RunPod serverless endpoint.
    Scales to zero when idle, bills per second of GPU use.

    Requires:
        - RUNPOD_API_KEY
        - RUNPOD_ENDPOINT_ID (pre-configured RunPod serverless endpoint
          running the LiveTalking Docker image)
    """

    def __init__(self, api_key: str, endpoint_id: str) -> None:
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self._base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self._running = False
        self._client = httpx.AsyncClient(
            timeout=120.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def start(self) -> str:
        logger.info("Starting RunPod GPU endpoint: %s", self.endpoint_id)
        if await self.health_check():
            self._running = True
            return self._base_url
        raise RuntimeError(
            f"RunPod endpoint {self.endpoint_id} is not healthy. "
            "Check the RunPod dashboard for status."
        )

    async def stop(self) -> None:
        self._running = False
        await self._client.aclose()

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            data = resp.json()
            workers = data.get("workers", {})
            return workers.get("ready", 0) > 0 or workers.get("running", 0) > 0
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        return self._running
