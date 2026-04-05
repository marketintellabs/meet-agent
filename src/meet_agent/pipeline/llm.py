"""LLM provider — any OpenAI-compatible chat completion API."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


class LLMProvider:
    """Wraps any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.deepinfra.com/v1/openai",
        model: str = "nvidia/Nemotron-Mini-4B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str = "",
        max_history_turns: int = 50,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self._history: list[Message] = []
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def add_user_message(self, text: str, speaker: str = "Participant") -> None:
        """Add a transcribed user utterance to the conversation history."""
        content = f"[{speaker}]: {text}" if speaker else text
        self._history.append(Message(role="user", content=content))
        self._trim_history()

    def add_assistant_message(self, text: str) -> None:
        """Add the agent's own response to the conversation history."""
        self._history.append(Message(role="assistant", content=text))
        self._trim_history()

    async def generate_response(self, user_text: str | None = None) -> str:
        """Generate a response from the LLM given the current conversation history."""
        if user_text:
            self.add_user_message(user_text)

        messages = self._build_messages()
        url = f"{self.api_base}/chat/completions"

        resp = await self._client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"].strip()
        self.add_assistant_message(text)
        logger.debug("LLM response: %s", text[:200])
        return text

    def clear_history(self) -> None:
        self._history.clear()

    def _build_messages(self) -> list[Message]:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.extend(self._history)
        return messages

    def _trim_history(self) -> None:
        if len(self._history) > self.max_history_turns * 2:
            self._history = self._history[-(self.max_history_turns * 2) :]
