# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
python main.py
```

Requires a running Ollama server. The endpoint is hardcoded to `http://192.168.178.70:11434` — update this if running on a different machine.

## Architecture

This is a proof-of-concept demonstrating **LLM chaining** with LangChain + Ollama. Two local models are chained sequentially:

1. **LLM 1** (`gpt-oss:20b`) — Translates user input from English to French
2. **LLM 2** (`gemma4:31b`) — Receives LLM 1's output + a weather query, translates to German, and can call the `get_weather` tool

The chain flow:
```
"I love programming." → [LLM1: English→French] → output + weather query → [LLM2: English→German + tool use] → final output
```

Tool integration uses LangChain's `@tool` decorator; the weather tool is a stub. LLM2 has the tool bound via `llm2.bind_tools([get_weather])`.

No tests, no build system, no linter configured. Dependencies live in `.venv/`.
