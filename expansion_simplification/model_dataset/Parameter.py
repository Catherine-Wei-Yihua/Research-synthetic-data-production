#!/usr/bin/env python3
"""Configure Together AI chat-completion hyperparameters.

This module is intentionally independent of file conversion and API calls. The
future Batch converter can import :func:`build_request_parameters` and use its
return value as the common parameter block for every request.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# Editable defaults. Keep these names aligned with Together's request fields so
# the configuration is easy to scan and modify.
model: str | None = None
temperature = 0.2
max_tokens = 5_000
reasoning_effort = "low"
reasoning = {"enabled": False}
response_format = {"type": "json_object"}

REASONING_EFFORT_VALUES = ("low", "medium", "high", "max", "xhigh")

# Together's public serverless model IDs are listed explicitly so unsupported
# request fields can be omitted instead of being sent speculatively. Extend
# these sets when Together adds or renames models.
REASONING_EFFORT_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-V4-Pro-0813",
    }
)

REASONING_TOGGLE_MODELS = frozenset(
    {
        "MiniMaxAI/MiniMax-M3",
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-V4-Pro-0813",
        "deepcogito/cogito-v2-1-671b",
        "moonshotai/Kimi-K2.5",
        "moonshotai/Kimi-K2.6",
        "moonshotai/Kimi-K3",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.6-Plus",
        "zai-org/GLM-5",
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.2",
    }
)


@dataclass(frozen=True)
class ModelCapabilities:
    """Reasoning controls supported by one model."""

    reasoning_effort: bool = False
    reasoning_toggle: bool = False


@dataclass(frozen=True)
class Hyperparameters:
    """Validated values used to construct a Together chat request."""

    model: str
    temperature: float = temperature
    max_tokens: int = max_tokens
    reasoning_effort: str | None = None
    reasoning_enabled: bool | None = None
    response_format: Mapping[str, Any] = field(
        default_factory=lambda: dict(response_format)
    )

    def to_request_parameters(self) -> dict[str, Any]:
        """Return only the parameters supported by the selected model."""
        return build_request_parameters(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            reasoning_enabled=self.reasoning_enabled,
            response_format=self.response_format,
        )


def capabilities_for_model(model: str) -> ModelCapabilities:
    """Look up supported reasoning controls for a Together model ID."""
    normalized_model = model.strip().casefold()
    effort_models = {item.casefold() for item in REASONING_EFFORT_MODELS}
    toggle_models = {item.casefold() for item in REASONING_TOGGLE_MODELS}
    return ModelCapabilities(
        reasoning_effort=normalized_model in effort_models,
        reasoning_toggle=normalized_model in toggle_models,
    )


def default_reasoning_effort() -> str:
    """Return the editable default without conflicting with a function argument."""
    return reasoning_effort


def default_reasoning_enabled() -> bool:
    """Return the editable default reasoning toggle."""
    return bool(reasoning["enabled"])


def default_response_format() -> dict[str, Any]:
    """Return a fresh copy of the editable response-format default."""
    return dict(response_format)


def build_request_parameters(
    *,
    model: str,
    temperature: float = temperature,
    max_tokens: int = max_tokens,
    reasoning_effort: str | None = None,
    reasoning_enabled: bool | None = None,
    response_format: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a model-aware parameter dictionary for Together Batch requests.

    ``reasoning_effort`` defaults to ``"low"`` only for models that support
    adjustable effort. ``reasoning`` defaults to ``{"enabled": false}`` only
    for hybrid models. Both fields are omitted for unsupported models.
    """
    model = model.strip()
    if not model:
        raise ValueError("model cannot be empty")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    selected_response_format = dict(response_format or default_response_format())
    if not selected_response_format.get("type"):
        raise ValueError("response_format must contain a non-empty 'type' field")

    capabilities = capabilities_for_model(model)
    parameters: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": selected_response_format,
    }

    if capabilities.reasoning_effort:
        selected_effort = reasoning_effort or default_reasoning_effort()
        if selected_effort not in REASONING_EFFORT_VALUES:
            allowed = ", ".join(REASONING_EFFORT_VALUES)
            raise ValueError(f"reasoning_effort must be one of: {allowed}")
        if model.casefold().startswith("openai/gpt-oss-") and selected_effort not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("GPT-OSS reasoning_effort must be low, medium, or high")
        parameters["reasoning_effort"] = selected_effort
    elif reasoning_effort is not None:
        raise ValueError(f"Model {model!r} does not support reasoning_effort")

    if capabilities.reasoning_toggle:
        selected_reasoning = (
            default_reasoning_enabled()
            if reasoning_enabled is None
            else reasoning_enabled
        )
        parameters["reasoning"] = {"enabled": selected_reasoning}
    elif reasoning_enabled is not None:
        raise ValueError(f"Model {model!r} does not support the reasoning toggle")

    return parameters


def parse_response_format(value: str) -> dict[str, Any]:
    """Parse a response-format shorthand or JSON object from the CLI."""
    if value in {"json_object", "text"}:
        return {"type": value}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "response format must be 'json_object', 'text', or a JSON object"
        ) from error
    if not isinstance(parsed, dict) or not parsed.get("type"):
        raise argparse.ArgumentTypeError(
            "response format JSON must be an object with a non-empty 'type' field"
        )
    return parsed


def reasoning_choice(value: str) -> bool:
    """Convert an enabled/disabled CLI value to a boolean."""
    choices = {"enabled": True, "disabled": False, "true": True, "false": False}
    try:
        return choices[value.casefold()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "reasoning must be enabled, disabled, true, or false"
        ) from error


def add_hyperparameter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared Together hyperparameter options to a CLI parser."""
    parser.add_argument(
        "--model",
        default=model,
        required=model is None,
        help="Together model ID, for example openai/gpt-oss-20b",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=temperature,
        help=f"Sampling temperature (default: {temperature})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=max_tokens,
        help=f"Maximum output tokens (default: {max_tokens})",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Reasoning effort for supported models; supported-model default: "
            f"{reasoning_effort}"
        ),
    )
    parser.add_argument(
        "--reasoning",
        type=reasoning_choice,
        default=None,
        metavar="{enabled,disabled}",
        help="Reasoning toggle for hybrid models (supported-model default: disabled)",
    )
    parser.add_argument(
        "--response-format",
        type=parse_response_format,
        default=dict(response_format),
        metavar="FORMAT",
        help=(
            "Response format shorthand or JSON object "
            "(default: '{\"type\":\"json_object\"}')"
        ),
    )


def hyperparameters_from_namespace(namespace: argparse.Namespace) -> Hyperparameters:
    """Build a Hyperparameters instance from shared CLI options."""
    return Hyperparameters(
        model=namespace.model,
        temperature=namespace.temperature,
        max_tokens=namespace.max_tokens,
        reasoning_effort=namespace.reasoning_effort,
        reasoning_enabled=namespace.reasoning,
        response_format=namespace.response_format,
    )


def parse_args(argv: Sequence[str] | None = None) -> Hyperparameters:
    """Parse manually adjustable hyperparameters from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_hyperparameter_arguments(parser)
    namespace = parser.parse_args(argv)
    return hyperparameters_from_namespace(namespace)


def main() -> None:
    """Print the resolved request parameters for inspection or scripting."""
    try:
        parameters = parse_args().to_request_parameters()
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error
    print(json.dumps(parameters, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
