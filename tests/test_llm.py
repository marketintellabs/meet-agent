"""Tests for LLM provider."""

from meet_agent.pipeline.llm import LLMProvider, Message


def test_history_management():
    llm = LLMProvider(api_key="test", max_history_turns=5)
    for i in range(20):
        llm.add_user_message(f"message {i}")
    assert len(llm.history) == 10  # max_history_turns * 2


def test_build_messages_with_system_prompt():
    llm = LLMProvider(api_key="test", system_prompt="You are helpful.")
    llm.add_user_message("Hello")
    messages = llm._build_messages()
    assert messages[0].role == "system"
    assert messages[0].content == "You are helpful."
    assert messages[1].role == "user"


def test_clear_history():
    llm = LLMProvider(api_key="test")
    llm.add_user_message("Hello")
    llm.add_assistant_message("Hi")
    llm.clear_history()
    assert len(llm.history) == 0


def test_user_message_with_speaker():
    llm = LLMProvider(api_key="test")
    llm.add_user_message("Hello", speaker="Alice")
    assert "[Alice]" in llm.history[0].content
