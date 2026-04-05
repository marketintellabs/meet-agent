# MeetAgent

**Open-source AI agents for video meetings. Google Meet. Zoom. Your LLM.**

MeetAgent lets AI agents join Google Meet and Zoom calls as real participants — listening to the conversation, reasoning with any LLM, and speaking back with synthesized voice. No vendor lock-in, fully self-hostable, and 10-30x cheaper than proprietary alternatives.

## Features

- **Google Meet + Zoom** support out of the box
- **Any LLM** — DeepInfra, OpenAI, Anthropic, Ollama, or any OpenAI-compatible API
- **Pluggable STT** — DeepInfra Whisper, OpenAI Whisper, or local faster-whisper (free)
- **Pluggable TTS** — OpenAI TTS or local Piper (free)
- **Voice Activity Detection** — silero-vad with energy-based fallback
- **REST API** — create and manage sessions programmatically
- **CLI** — join meetings with a single command
- **Docker** — one command to run everything
- **Avatar rendering** (v0.2) — real-time lip-synced avatar via LiveTalking

## Quick Start

### Install

```bash
pip install meet-agent
playwright install chromium
```

### Join a Meeting

```bash
# Set your API key
export LLM_API_KEY=your-deepinfra-or-openai-key

# Join a Google Meet
meet-agent join "https://meet.google.com/abc-defg-hij"

# Join a Zoom meeting
meet-agent join "https://zoom.us/j/123456789" --name "AI Assistant"

# With a custom persona
meet-agent join "https://meet.google.com/abc-defg-hij" \
  --name "Financial Analyst" \
  --system-prompt "You are a senior financial analyst. Keep responses concise."
```

### Docker

```bash
# Clone the repo
git clone https://github.com/MarketIntelLabs/meet-agent.git
cd meet-agent

# Set your API keys
cp .env.example .env
# Edit .env with your keys

# Run
docker compose up
```

### API Server

```bash
# Start the server
meet-agent serve --port 8080

# Create a session via API
curl -X POST http://localhost:8080/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://meet.google.com/abc-defg-hij",
    "agent_name": "My Agent",
    "system_prompt": "You are a helpful meeting assistant."
  }'

# Get session status
curl http://localhost:8080/sessions/{session_id}

# Get transcript
curl http://localhost:8080/sessions/{session_id}/transcript

# Stop a session
curl -X POST http://localhost:8080/sessions/{session_id}/stop
```

## Architecture

```
Meeting (Google Meet / Zoom)
    |
    v
Headless Chromium (Playwright)
    |
    +---> Audio Capture ---> VAD ---> STT ---> LLM ---> TTS ---> Audio Output
    |                      (silero)  (Whisper) (any)   (OpenAI)      |
    +<--------------------------------------------------------------|
```

MeetAgent uses a headless Chromium browser to join meetings — the same approach used by humans. This means it works with any browser-based meeting platform without needing special APIs or SDKs.

### Processing Pipeline

1. **Audio Capture** — Browser captures meeting audio via Web Audio API
2. **VAD** — Voice Activity Detection segments continuous audio into speech utterances
3. **STT** — Speech-to-Text transcribes each utterance
4. **LLM** — Generates a contextual response based on conversation history
5. **TTS** — Text-to-Speech synthesizes the response as audio
6. **Audio Output** — Plays the audio back into the meeting

### Session State Machine

```
IDLE --> LISTENING --> THINKING --> SPEAKING --> LISTENING --> ...
                                       |
                                       v
                                    STOPPED
```

## Configuration

All configuration is via environment variables (or `.env` file):

### LLM (Required)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE` | `https://api.deepinfra.com/v1/openai` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | — | API key |
| `LLM_MODEL` | `nvidia/Nemotron-Mini-4B-Instruct` | Model name |
| `LLM_MAX_TOKENS` | `512` | Max response tokens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |

### STT (Speech-to-Text)

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_PROVIDER` | `deepinfra` | `deepinfra`, `openai`, or `faster-whisper` |
| `STT_API_KEY` | (falls back to `LLM_API_KEY`) | API key for STT provider |
| `STT_MODEL` | `openai/whisper-large-v3-turbo` | Whisper model |
| `STT_LANGUAGE` | `en` | Language code |

### TTS (Text-to-Speech)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PROVIDER` | `openai` | `openai` or `piper` |
| `TTS_API_KEY` | (falls back to `LLM_API_KEY`) | API key for TTS provider |
| `TTS_VOICE` | `alloy` | Voice selection |
| `TTS_SPEED` | `1.0` | Speech speed multiplier |

### Agent

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_NAME` | `MeetAgent` | Display name in the meeting |
| `AGENT_SYSTEM_PROMPT` | (helpful assistant) | System prompt for the LLM |

### VAD (Voice Activity Detection)

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_THRESHOLD` | `0.5` | Speech detection threshold (0.0-1.0) |
| `VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration to trigger |
| `VAD_MIN_SILENCE_MS` | `700` | Silence duration to end a segment |

## Provider Combinations

### Cheapest (All Local — Free)

```bash
STT_PROVIDER=faster-whisper
TTS_PROVIDER=piper
LLM_API_BASE=http://localhost:11434/v1  # Ollama
LLM_MODEL=llama3.2
```

### Recommended (Best Quality/Cost Balance)

```bash
LLM_API_BASE=https://api.deepinfra.com/v1/openai
LLM_API_KEY=your-key
LLM_MODEL=nvidia/Nemotron-Mini-4B-Instruct
STT_PROVIDER=deepinfra    # $0.0002/min
TTS_PROVIDER=openai        # $0.015/1K chars
TTS_API_KEY=your-openai-key
```

### Highest Quality

```bash
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=your-key
LLM_MODEL=gpt-4o
STT_PROVIDER=openai
TTS_PROVIDER=openai
TTS_VOICE=nova
```

## Cost Comparison

| | MeetAgent (API) | MeetAgent (Self-Hosted) | PikaStream |
|---|---|---|---|
| 30-min meeting | ~$0.50-1.00 | Free | $8-15 |
| STT | $0.006 | Free | Included |
| LLM | $0.05-0.50 | Free (Ollama) | Included |
| TTS | $0.22 | Free (Piper) | Included |

## Roadmap

### v0.1 (Current) — Voice Agent

- [x] Google Meet connector
- [x] Zoom connector
- [x] Pluggable STT (DeepInfra, OpenAI, faster-whisper)
- [x] Pluggable TTS (OpenAI, Piper)
- [x] OpenAI-compatible LLM provider
- [x] Voice Activity Detection
- [x] Session orchestrator
- [x] CLI + REST API
- [x] Docker support

### v0.2 — Animated Avatar

- [ ] Real-time lip-synced avatar via LiveTalking
- [ ] GPU provider abstraction (local, RunPod, AWS EC2)
- [ ] Voice cloning support
- [ ] Static image fallback (no GPU)

### v0.3+ — Ecosystem

- [ ] Microsoft Teams connector
- [ ] Hermes/Paperclip skill integration
- [ ] Meeting memory (cross-session context)
- [ ] MCP tool execution during calls
- [ ] Web dashboard for session management
- [ ] skills.sh marketplace listing

## Examples

See the [`examples/`](./examples/) directory:

- **`basic_assistant.py`** — Simple helpful meeting assistant
- **`financial_analyst.py`** — MarketIntelLabs financial analyst demo

## Development

```bash
# Clone
git clone https://github.com/MarketIntelLabs/meet-agent.git
cd meet-agent

# Install with dev dependencies
pip install -e ".[dev,all-local]"
playwright install chromium

# Run tests
pytest -v

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Run `ruff check` and `pytest` before submitting
5. Open a PR with a clear description

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

Built by [MarketIntelLabs](https://marketintellabs.com) — AI-powered financial intelligence.
