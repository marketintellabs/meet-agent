"""Abstract GPU provider interface for avatar rendering backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GPUProvider(ABC):
    """Manages GPU compute resources for avatar rendering.

    Abstracts away the infrastructure so the avatar renderer
    can work with any GPU backend: local, RunPod, AWS EC2, etc.
    """

    @abstractmethod
    async def start(self) -> str:
        """Start or connect to the GPU backend.

        Returns:
            The base URL of the running avatar rendering server.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the GPU backend and release resources."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the GPU backend is healthy and ready."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool: ...
