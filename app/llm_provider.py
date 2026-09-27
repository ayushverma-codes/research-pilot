"""
Thin, swappable LLM wrapper.

The rest of the app never calls a provider SDK directly - it calls
`get_llm_client().complete(prompt, system=...)`. This keeps the provider
configurable via .env (LLM_PROVIDER, LLM_MODEL, and the relevant API key)
without touching agent logic. Anthropic and Groq are implemented; adding a
new provider means adding one small class here.
"""

from __future__ import annotations

import os
import time
import threading
import queue
from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


def _llm_timeout_seconds() -> float:
    """Return a positive, bounded provider request timeout from the environment."""
    raw = os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS)).strip()
    try:
        timeout = float(raw)
    except ValueError as e:
        raise LLMError(
            f"LLM_TIMEOUT_SECONDS must be a positive number of seconds, got {raw!r}."
        ) from e
    if timeout <= 0:
        raise LLMError("LLM_TIMEOUT_SECONDS must be greater than 0.")
    return timeout


class LLMError(RuntimeError):
    """Raised when the LLM call fails or is misconfigured."""




def _run_with_hard_timeout(call, timeout_seconds: float):
    """Run a provider SDK call behind a wall-clock timeout.

    Provider/http-client timeouts are still configured as the first line of
    defence, but they can cover individual socket phases rather than the whole
    call on every SDK/version.  This daemon-thread guard guarantees the CLI
    regains control after ``timeout_seconds`` even if the SDK itself wedges.
    """
    results: queue.Queue = queue.Queue(maxsize=1)

    def runner():
        try:
            results.put((True, call()))
        except BaseException as exc:  # propagate provider exceptions to caller
            results.put((False, exc))

    worker = threading.Thread(target=runner, daemon=True, name="researchpilot-llm-call")
    worker.start()
    try:
        ok, value = results.get(timeout=timeout_seconds)
    except queue.Empty as e:
        raise LLMError(
            f"LLM call exceeded the {timeout_seconds:g}s hard timeout."
        ) from e

    if ok:
        return value
    raise value


class RateLimiter:
    """
    Simple client-side throttle: guarantees at least `min_interval` seconds
    between calls, sized from a requests-per-minute budget. This is a
    proactive guard (spread requests out) that runs *in addition to* the
    reactive retry-after handling below - it exists to keep normal usage
    from bumping into the limit in the first place, not to replace the
    server's own accounting.
    """

    def __init__(self, requests_per_minute: float):
        self._min_interval = 60.0 / max(requests_per_minute, 1)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


class AnthropicClient:
    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError(
                "The 'anthropic' package is required. Install it with "
                "`pip install anthropic`."
            ) from e

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and add your key."
            )
        # Disable SDK-level retries so one provider stall cannot silently turn
        # into several long waits. Higher-level agent nodes already own their
        # bounded retry/fallback behavior.
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=_llm_timeout_seconds(),
            max_retries=0,
        )
        self.model = model

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1500) -> str:
        try:
            timeout = _llm_timeout_seconds()
            response = _run_with_hard_timeout(
                lambda: self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout,
            )
        except Exception as e:  # noqa: BLE001 - surface as a single app-level error
            raise LLMError(f"LLM call failed: {e}") from e

        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "".join(text_parts).strip()


class GroqClient:
    """
    Groq chat-completions client with:
      - a proactive rate limiter (min interval between calls, from
        GROQ_REQUESTS_PER_MINUTE, default 25 - conservative vs. Groq's
        common free-tier ~30 RPM so normal use shouldn't ever hit a 429)
      - reactive handling of 429s: reads the server's retry-after and
        sleeps that long before a single bounded retry, instead of
        failing immediately or hammering the API.
    """

    def __init__(self, model: str, requests_per_minute: float = 25, max_retries: int = 3):
        try:
            import groq
        except ImportError as e:
            raise LLMError(
                "The 'groq' package is required. Install it with `pip install groq`."
            ) from e

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        self._groq = groq
        # max_retries=0 here: our own loop below owns retry/backoff decisions
        # (so we can rate-limit-aware sleep on the exact retry-after value)
        # instead of letting the SDK retry silently with its own backoff.
        self._client = groq.Groq(
            api_key=api_key,
            max_retries=0,
            timeout=_llm_timeout_seconds(),
        )
        self.model = model
        self.max_retries = max_retries
        self._limiter = RateLimiter(requests_per_minute)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1500) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                timeout = _llm_timeout_seconds()
                response = _run_with_hard_timeout(
                    lambda: self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                    ),
                    timeout,
                )
                return (response.choices[0].message.content or "").strip()
            except self._groq.RateLimitError as e:
                last_error = e
                retry_after = self._parse_retry_after(e, default=5.0)
                if attempt < self.max_retries:
                    time.sleep(retry_after)
                    continue
                raise LLMError(
                    f"Groq rate limit exceeded after {self.max_retries} retries: {e}"
                ) from e
            except self._groq.APIError as e:
                # Non-rate-limit API errors are not retried here.
                raise LLMError(f"Groq LLM call failed: {e}") from e

        raise LLMError(f"Groq LLM call failed after retries: {last_error}")

    @staticmethod
    def _parse_retry_after(error: Exception, default: float) -> float:
        """Best-effort extraction of the retry-after header (seconds) from a Groq error."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if headers:
            value = headers.get("retry-after")
            if value:
                try:
                    return float(value)
                except ValueError:
                    pass
        return default


def get_llm_client():
    """Factory that returns the configured LLM client based on .env settings."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        return AnthropicClient(model=model)

    if provider == "groq":
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        rpm = float(os.getenv("GROQ_REQUESTS_PER_MINUTE", "25"))
        return GroqClient(model=model, requests_per_minute=rpm)

    raise LLMError(f"Unsupported LLM_PROVIDER: '{provider}'. Supported: anthropic, groq")
