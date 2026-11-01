"""OpenAI-compatible model API wrapper for the TSSDA data agent.

This module owns model configuration, OpenAI-compatible client
initialization, API calls, and JSON response parsing. Prompt construction and
runtime orchestration remain in the semantic and SQL generation agents.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .infrastructure import (
    InsufficientBalanceError,
    insufficient_balance_error_reason,
)


DEFAULT_DEEPSEEK_MODEL = ""
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_QWEN_MODEL = ""
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RETRY_TOKEN_MULTIPLIER = 2


class EmptyContentError(ValueError):
    """Raised when one model attempt returns no assistant content."""


class DeepSeekEmptyContentError(EmptyContentError):
    """Raised when one DeepSeek attempt returns no assistant content."""


class QwenEmptyContentError(EmptyContentError):
    """Raised when one Qwen attempt returns no assistant content."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class AgentConfig:
    """Configuration for an OpenAI-compatible model used by TSSDA."""

    provider: str = "deepseek"
    model: str = DEFAULT_DEEPSEEK_MODEL
    temperature: float | None = None
    max_output_tokens: int | None = 16384
    api_key: str | None = None
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    reasoning_effort: str | None = None
    thinking_enabled: bool = True
    api_key_env_name: str = "DEEPSEEK_API_KEY"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create a DeepSeek configuration from environment variables."""

        temperature_value = os.getenv("DEEPSEEK_TEMPERATURE")
        max_tokens_value = os.getenv("DEEPSEEK_MAX_TOKENS", "16384")
        return cls(
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            temperature=(
                float(temperature_value)
                if temperature_value not in {None, ""}
                else None
            ),
            max_output_tokens=(
                int(max_tokens_value) if max_tokens_value else None
            ),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT") or None,
            thinking_enabled=_env_bool("DEEPSEEK_THINKING_ENABLED", True),
            api_key_env_name="DEEPSEEK_API_KEY",
        )

    @classmethod
    def from_qwen_env(cls) -> "AgentConfig":
        """Create a Qwen configuration from environment variables."""

        temperature_value = os.getenv("QWEN_TEMPERATURE")
        max_tokens_value = os.getenv("QWEN_MAX_TOKENS", "16384")
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        return cls(
            provider="qwen",
            model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
            temperature=(
                float(temperature_value)
                if temperature_value not in {None, ""}
                else None
            ),
            max_output_tokens=(
                int(max_tokens_value) if max_tokens_value else None
            ),
            api_key=api_key,
            base_url=os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
            reasoning_effort=(
                os.getenv("QWEN_REASONING_EFFORT") or None
            ),
            thinking_enabled=_env_bool("QWEN_THINKING_ENABLED", True),
            api_key_env_name=(
                "QWEN_API_KEY"
                if os.getenv("QWEN_API_KEY")
                else "DASHSCOPE_API_KEY"
            ),
        )


class DataAgent:
    """Minimal OpenAI-compatible Chat Completions caller used by TSSDA."""

    def __init__(self, config: AgentConfig | None = None, client: Any | None = None):
        self.config = config or AgentConfig.from_env()
        self._client = client
        self._response_diagnostics_local = threading.local()
        self._state_lock = threading.Lock()
        self._call_count = 0
        self._usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "responses_with_prompt_tokens": 0,
        }

    def consume_response_diagnostics(self) -> list[dict[str, Any]]:
        """Return and clear response metadata for the current worker thread."""

        diagnostics = list(self._thread_response_diagnostics())
        self._thread_response_diagnostics().clear()
        return diagnostics

    def usage_snapshot(self) -> dict[str, int]:
        """Return cumulative provider token usage without resetting it."""

        with self._state_lock:
            return dict(self._usage_totals)

    def _thread_response_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = getattr(self._response_diagnostics_local, "items", None)
        if diagnostics is None:
            diagnostics = []
            self._response_diagnostics_local.items = diagnostics
        return diagnostics

    @contextmanager
    def temporary_max_output_tokens(self, value: int | None):
        """Temporarily override the per-request output-token limit."""

        previous = self.config.max_output_tokens
        self.config.max_output_tokens = value
        try:
            yield
        finally:
            self.config.max_output_tokens = previous

    @contextmanager
    def temporary_thinking_enabled(self, value: bool):
        """Temporarily override provider thinking mode."""

        previous = self.config.thinking_enabled
        self.config.thinking_enabled = value
        try:
            yield
        finally:
            self.config.thinking_enabled = previous

    @property
    def client(self) -> Any:
        """Lazily initialize the OpenAI-compatible provider client."""

        if self._client is None:
            if not self.config.api_key:
                raise RuntimeError(
                    f"{self.config.api_key_env_name} is not configured for "
                    f"provider {self.config.provider!r}."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The OpenAI-compatible Python SDK is required. "
                    "Install it with `pip install openai`."
                ) from exc

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call the configured provider and return assistant plain text."""

        if not (model or self.config.model):
            raise ValueError("Configure the provider model identifier before calling the agent")
        request: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": False,
        }
        selected_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        if selected_temperature is not None:
            request["temperature"] = selected_temperature
        selected_max_tokens = (
            self.config.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        if selected_max_tokens is not None:
            request["max_tokens"] = selected_max_tokens
        if response_format is not None:
            request["response_format"] = dict(response_format)
        if self.config.reasoning_effort and self.config.provider == "deepseek":
            request["reasoning_effort"] = self.config.reasoning_effort
        selected_thinking_enabled = (
            self.config.thinking_enabled
            if thinking_enabled is None
            else thinking_enabled
        )
        with self._state_lock:
            self._call_count += 1
            call_index = self._call_count
        attempt_request = dict(request)
        if self.config.provider == "qwen":
            qwen_extra_body: dict[str, Any] = {
                "enable_thinking": selected_thinking_enabled
            }
            if selected_thinking_enabled and self.config.reasoning_effort:
                qwen_extra_body["reasoning_effort"] = (
                    self.config.reasoning_effort
                )
            attempt_request["extra_body"] = qwen_extra_body
            attempt_request.pop("reasoning_effort", None)
        elif self.config.provider == "deepseek":
            attempt_request["extra_body"] = {
                "thinking": {
                    "type": (
                        "enabled" if selected_thinking_enabled else "disabled"
                    )
                }
            }
            if not selected_thinking_enabled:
                attempt_request.pop("reasoning_effort", None)

        try:
            response = self.client.chat.completions.create(**attempt_request)
        except InsufficientBalanceError:
            raise
        except Exception as exc:
            balance_reason = insufficient_balance_error_reason(exc)
            if balance_reason:
                raise InsufficientBalanceError(balance_reason) from exc
            raise
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError(
                f"{self.config.provider} returned no completion choices"
            )
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = self._assistant_content(message)
        self._record_response_diagnostic(
            response=response,
            choice=choice,
            content=content,
            call_index=call_index,
            response_attempt=1,
            model=str(attempt_request["model"]),
            max_tokens=selected_max_tokens,
            thinking_enabled=selected_thinking_enabled,
            response_format=response_format,
        )
        if content:
            return content

        error_class = (
            QwenEmptyContentError
            if self.config.provider == "qwen"
            else DeepSeekEmptyContentError
        )
        raise error_class(
            f"{self.config.provider} returned empty assistant content"
        )

    @staticmethod
    def _assistant_content(message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))
                elif getattr(part, "text", None):
                    text_parts.append(str(part.text))
            return "".join(text_parts).strip()
        return ""

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    def _record_response_diagnostic(
        self,
        *,
        response: Any,
        choice: Any,
        content: str,
        call_index: int,
        response_attempt: int,
        model: str,
        max_tokens: int | None,
        thinking_enabled: bool,
        response_format: dict[str, str] | None,
    ) -> None:
        usage = self._field(response, "usage")
        prompt_tokens = self._usage_integer(
            self._field(usage, "prompt_tokens")
            if self._field(usage, "prompt_tokens") is not None
            else self._field(usage, "input_tokens")
        )
        completion_tokens = self._usage_integer(
            self._field(usage, "completion_tokens")
            if self._field(usage, "completion_tokens") is not None
            else self._field(usage, "output_tokens")
        )
        total_tokens = self._usage_integer(self._field(usage, "total_tokens"))
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        completion_details = self._field(usage, "completion_tokens_details")
        reasoning_tokens = self._field(completion_details, "reasoning_tokens")
        if isinstance(reasoning_tokens, bool) or not isinstance(
            reasoning_tokens, int
        ):
            reasoning_tokens = None
        self._thread_response_diagnostics().append(
            {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "provider": self.config.provider,
                "call_index": call_index,
                "response_attempt": response_attempt,
                "model": str(self._field(response, "model") or model),
                "thinking_enabled": thinking_enabled,
                "reasoning_effort": (
                    self.config.reasoning_effort
                    if thinking_enabled
                    else None
                ),
                "max_tokens": max_tokens,
                "response_format": (
                    response_format.get("type")
                    if isinstance(response_format, dict)
                    else None
                ),
                "finish_reason": self._field(choice, "finish_reason"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "content_length": len(content),
                "empty_content": not bool(content),
            }
        )
        with self._state_lock:
            if prompt_tokens is not None:
                self._usage_totals["prompt_tokens"] += prompt_tokens
                self._usage_totals["responses_with_prompt_tokens"] += 1
            if completion_tokens is not None:
                self._usage_totals["completion_tokens"] += completion_tokens
            if total_tokens is not None:
                self._usage_totals["total_tokens"] += total_tokens

    @staticmethod
    def _usage_integer(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def call_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_enabled: bool | None = None,
    ) -> str:
        """Call the provider with a simple system/user prompt pair."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.call(
            messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_enabled=thinking_enabled,
        )

    def call_json_value(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        retry_invalid_json: bool = True,
        response_format: dict[str, str] | None = None,
        thinking_enabled: bool | None = None,
    ) -> Any:
        """Call the provider and parse a top-level JSON value."""

        text = self.call(
            messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_enabled=thinking_enabled,
            response_format=response_format,
        )
        try:
            return self._parse_json(text)
        except ValueError:
            if not retry_invalid_json:
                raise
            retry_base_tokens = (
                max_output_tokens
                or self.config.max_output_tokens
                or 16384
            )
            retry_max_tokens = retry_base_tokens * RETRY_TOKEN_MULTIPLIER
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Return the answer again as valid JSON only. Do not "
                        "include markdown, comments, explanations, or any "
                        "text outside the JSON value."
                    ),
                },
            ]
            retry_text = self.call(
                retry_messages,
                model=model,
                temperature=temperature,
                max_output_tokens=retry_max_tokens,
                thinking_enabled=thinking_enabled,
                response_format=response_format,
            )
            return self._parse_json(retry_text)

    def call_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        retry_invalid_json: bool = True,
        thinking_enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Call the provider and require a JSON object response."""

        parsed = self.call_json_value(
            self._ensure_json_object_instruction(messages),
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retry_invalid_json=retry_invalid_json,
            thinking_enabled=thinking_enabled,
            response_format={"type": "json_object"},
        )
        if not isinstance(parsed, dict):
            raise ValueError("DataAgent.call_json expected a JSON object response")
        return parsed

    @staticmethod
    def _ensure_json_object_instruction(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Ensure JSON Object mode sees an explicit JSON instruction."""

        if any(
            "json" in str(message.get("content") or "").casefold()
            for message in messages
        ):
            return messages
        return [
            {
                "role": "system",
                "content": "Return exactly one valid JSON object.",
            },
            *messages,
        ]

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Parse JSON while tolerating an accidental Markdown code fence."""

        candidate = text.strip().lstrip("\ufeff")
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].strip().lower() in {
                "```",
                "```json",
                "```javascript",
            }:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            for index, char in enumerate(candidate):
                if char not in "{[":
                    continue
                try:
                    value, _ = decoder.raw_decode(candidate[index:])
                    return value
                except json.JSONDecodeError:
                    continue
            raise ValueError(
                f"Model response is not valid JSON: {candidate[:200]!r}"
            ) from exc


def create_default_agent() -> DataAgent:
    """Create a data agent from environment configuration."""

    return DataAgent(AgentConfig.from_env())


def create_deepseek_agent(config: AgentConfig | None = None) -> DataAgent:
    """Create an explicitly named DeepSeek-backed data agent."""

    return DataAgent(config or AgentConfig.from_env())


def create_qwen_agent(config: AgentConfig | None = None) -> DataAgent:
    """Create a Qwen-backed data agent using QWEN_* environment settings."""

    return DataAgent(config or AgentConfig.from_qwen_env())


DeepSeekDataAgent = DataAgent
QwenDataAgent = DataAgent

# Compatibility alias for existing imports. The implementation now uses the
# OpenAI-compatible DeepSeek endpoint rather than OpenAI's Responses API.
OpenAIDataAgent = DataAgent
