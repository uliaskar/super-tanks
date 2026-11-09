"""
backend.py — OpenAI-compatible LLM factory for the Super Tanks / AgentDojo harness.

Reads its configuration from the environment (or a local harness/.env you create
from harness/.env.example):

  OPENAI_API_KEY   (required unless using Ollama) — your provider key. NEVER commit it.
  OPENAI_BASE_URL  (optional) — OpenAI-compatible endpoint.
                                default: https://api.openai.com/v1
                                Kimi/Moonshot: https://api.moonshot.ai/v1
                                OpenRouter:    https://openrouter.ai/api/v1
  OPENAI_MODEL     (optional) — model id. default: gpt-4o-mini

Ollama (local server, or the university OpenWebUI-fronted cluster) — set ONE of
these two URLs to a full OpenAI-compatible chat-completions base (the part
before "/chat/completions"):
  OLLAMA_URL         (optional) — university endpoint, e.g. https://openwebui.ux.uis.no/api/v1
  OLLAMA_TOKEN       (optional) — bearer token for OLLAMA_URL (local usually needs none)
  OLLAMA_MODEL       (optional) — model id for the university endpoint
  OLLAMA_LOCAL_URL   (optional) — local Ollama server, e.g. http://localhost:11434/v1
  OLLAMA_LOCAL_MODEL (optional) — model id for the local server

If OLLAMA_URL is set it wins (university cluster); otherwise OLLAMA_LOCAL_URL is
used if set (local server); otherwise the harness falls back to OPENAI_*. Both
Ollama routes are plain OpenAI-compatible chat completions — no custom request
translation.

It builds a plain `openai.OpenAI` client and wraps it in agentdojo's stock
`OpenAILLM`. We bypass agentdojo's get_llm/from_config and pass the OpenAILLM
instance straight into the pipeline, so no model needs to be registered in
agentdojo's ModelsEnum.

The API key/token is read at runtime and is never printed.

Kimi/Moonshot note: Moonshot rejects two things OpenAI tolerates — the OpenAI o1+
role name 'developer', and empty text content on a tool-call turn. When the chosen
base_url is a Moonshot endpoint we wrap chat.completions.create to rewrite those,
and force temperature=1.0 (kimi models reject other values). Other providers are
left untouched.
"""

import os
from pathlib import Path

import openai
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:8b-instruct"
ENV_FILE = Path(__file__).with_name(".env")


def _load_dotenv(path: Path) -> None:
    """Tiny dependency-free .env parser. Only sets keys not already in os.environ
    (a real environment variable always wins over the file)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(ENV_FILE)

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL")
OLLAMA_TOKEN = os.getenv("OLLAMA_TOKEN")
USING_UNIVERSITY = bool(OLLAMA_URL)
USING_OLLAMA = bool(OLLAMA_URL or OLLAMA_LOCAL_URL)

if OLLAMA_URL:
    BASE_URL = OLLAMA_URL.rstrip("/")
elif OLLAMA_LOCAL_URL:
    BASE_URL = OLLAMA_LOCAL_URL.rstrip("/")
else:
    BASE_URL = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

if USING_UNIVERSITY:
    MODEL_ID = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
elif USING_OLLAMA:
    MODEL_ID = os.getenv("OLLAMA_LOCAL_MODEL") or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
else:
    MODEL_ID = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _is_moonshot(base_url: str) -> bool:
    return "moonshot" in (base_url or "")


def _sanitize_messages(msgs) -> None:
    """Rewrite the outgoing OpenAI payload so Moonshot/Kimi accepts it.

    1. role 'developer' (OpenAI o1+ system naming) -> 'system'.
    2. empty text content: when Kimi replies with tool_calls and no prose,
       agentdojo echoes that turn back with content="" (or None/[]), and Kimi
       answers 400 'text content is empty'. Replace any empty content with a
       single space so tool_calls survive.
    """
    if not msgs:
        return
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "developer":
            m["role"] = "system"
        c = m.get("content")
        if c is None or (isinstance(c, str) and c.strip() == ""):
            m["content"] = " "
        elif isinstance(c, (list, tuple)):
            if len(c) == 0:
                m["content"] = " "
            else:
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text")
                        if not (isinstance(t, str) and t.strip()):
                            part["text"] = " "


def _patch_developer_role(client: openai.OpenAI) -> openai.OpenAI:
    """Wrap chat.completions.create to sanitize each outgoing payload for Kimi."""
    orig_create = client.chat.completions.create

    def create(*args, **kwargs):
        _sanitize_messages(kwargs.get("messages"))
        return orig_create(*args, **kwargs)

    client.chat.completions.create = create
    return client


def make_client(require_api_key: bool = True) -> openai.OpenAI:
    if USING_OLLAMA:
        if USING_UNIVERSITY and not OLLAMA_TOKEN and require_api_key:
            raise RuntimeError(
                "OLLAMA_TOKEN is not set. Add the university cluster token to harness/.env."
            )
        # Local Ollama servers accept any placeholder key; the university
        # cluster needs the real bearer token.
        return openai.OpenAI(api_key=OLLAMA_TOKEN or "ollama", base_url=BASE_URL)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and require_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it, or copy harness/.env.example to "
            "harness/.env and put your key there. (The key is never printed.)"
        )
    api_key = api_key or "offline-validation-placeholder"
    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)
    if _is_moonshot(BASE_URL):
        client = _patch_developer_role(client)
    return client


def make_llm(model: str = MODEL_ID, require_api_key: bool = True) -> OpenAILLM:
    """Return an agentdojo OpenAILLM ready to drop into a pipeline.

    Moonshot/Kimi requires temperature=1.0; other providers use agentdojo's
    default temperature.
    """
    client = make_client(require_api_key=require_api_key)
    if _is_moonshot(BASE_URL):
        return OpenAILLM(client, model, temperature=1.0)
    return OpenAILLM(client, model)


if __name__ == "__main__":
    # Prints config WITHOUT the key, so you can confirm what would be used.
    print(f"base_url = {BASE_URL}")
    print(f"model    = {MODEL_ID}")
    if USING_UNIVERSITY:
        provider, key_set = "ollama-university", bool(OLLAMA_TOKEN)
    elif USING_OLLAMA:
        provider, key_set = "ollama-local", True
    else:
        provider, key_set = "openai-compatible", bool(os.getenv("OPENAI_API_KEY"))
    print(f"provider = {provider}")
    print(f"api_key  = {'set' if key_set else 'NOT set'}")
