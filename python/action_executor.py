from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


DEFAULT_WEBHOOK_URL = "https://jsonplaceholder.typicode.com/posts"
REQUEST_TIMEOUT_SECONDS = 5
MAX_RETRIES = 2


@dataclass
class ActionExecutionResult:
    ok: bool
    status_code: int | None
    message: str
    request_id: str
    raw_response: dict | str | None


def _build_payload(action: str, arguments: dict[str, Any], request_id: str) -> dict[str, Any]:
    return {
        "action": action,
        "arguments": arguments,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def execute_action(
    action: str,
    arguments: dict[str, Any],
    *,
    webhook_url: str = DEFAULT_WEBHOOK_URL,
    debug: bool = False,
) -> ActionExecutionResult:
    request_id = str(uuid.uuid4())
    payload = _build_payload(action=action, arguments=arguments, request_id=request_id)

    last_error = "unknown_error"
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            if debug:
                print(
                    f"[DEBUG][action_executor] request_id={request_id} mode=live "
                    f"attempt={attempt} endpoint={webhook_url} action={action}"
                )

            response = requests.post(
                webhook_url,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )

            response_body: dict[str, Any] | str | None
            try:
                response_body = response.json()
            except ValueError:
                response_body = response.text

            if 200 <= response.status_code < 300:
                message = "Action accepted"
                if isinstance(response_body, dict):
                    message = str(response_body.get("message", response_body.get("title", message)))
                elif isinstance(response_body, str) and response_body.strip():
                    message = response_body.strip()[:200]

                if debug:
                    print(
                        f"[DEBUG][action_executor] request_id={request_id} "
                        f"status={response.status_code} ok=true"
                    )

                return ActionExecutionResult(
                    ok=True,
                    status_code=response.status_code,
                    message=message,
                    request_id=request_id,
                    raw_response=response_body,
                )

            # Retry only on transient server-side errors.
            last_error = f"HTTP {response.status_code}"
            should_retry = response.status_code >= 500 and attempt <= MAX_RETRIES
            if debug:
                print(
                    f"[DEBUG][action_executor] request_id={request_id} "
                    f"status={response.status_code} retry={should_retry}"
                )
            if not should_retry:
                return ActionExecutionResult(
                    ok=False,
                    status_code=response.status_code,
                    message=last_error,
                    request_id=request_id,
                    raw_response=response_body,
                )

        except requests.RequestException as exc:
            last_error = str(exc)
            should_retry = attempt <= MAX_RETRIES
            if debug:
                print(
                    f"[DEBUG][action_executor] request_id={request_id} "
                    f"error={last_error} retry={should_retry}"
                )
            if not should_retry:
                break

        time.sleep(0.25 * attempt)

    return ActionExecutionResult(
        ok=False,
        status_code=None,
        message=last_error,
        request_id=request_id,
        raw_response=None,
    )
