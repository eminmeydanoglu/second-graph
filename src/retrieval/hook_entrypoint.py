"""UserPromptSubmit hook entrypoint for push recall."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import os
import sys
from typing import Any

from .pipeline import recall_markdown


def _read_hook_input() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {}


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        return "\n".join(chunks)
    return ""


def _extract_query(payload: dict[str, Any]) -> str:
    for key in ("prompt", "query", "text", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            extracted = _extract_message_text(message.get("content"))
            if extracted.strip():
                return extracted

    return ""


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _hook_output(additional_context: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context,
        }
    }


def main() -> int:
    payload = _read_hook_input()

    if not _bool_env("RECALL_PUSH_ENABLED", True):
        print(json.dumps(_hook_output("")))
        return 0

    query = _extract_query(payload)
    if not query.strip():
        print(json.dumps(_hook_output("")))
        return 0

    depth = int(os.environ.get("RECALL_PUSH_DEPTH", "1"))
    limit = int(os.environ.get("RECALL_PUSH_LIMIT", "10"))
    timeout_seconds = float(os.environ.get("RECALL_HOOK_TIMEOUT_SEC", "2.5"))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                recall_markdown,
                query,
                mode="push",
                depth=depth,
                limit=limit,
            )
            markdown = future.result(timeout=max(0.1, timeout_seconds))
    except TimeoutError:
        print(json.dumps(_hook_output("")))
        return 0
    except Exception:
        print(json.dumps(_hook_output("")))
        return 0

    print(json.dumps(_hook_output(markdown)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
