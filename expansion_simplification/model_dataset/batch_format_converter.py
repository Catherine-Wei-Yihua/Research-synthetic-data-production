#!/usr/bin/env python3
"""Convert raw question JSONL into Together AI Batch API request JSONL."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__:
    from .Parameter import add_hyperparameter_arguments, hyperparameters_from_namespace
else:
    from Parameter import add_hyperparameter_arguments, hyperparameters_from_namespace


DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompt.md")
REQUIRED_PROMPT_NAMES = ("COMMAND_PROMPT", "CONTENT_PROMPT", "TEMPLATE_PROMPT")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "batch"

MODEL_SHORT_NAMES = {
    "Qwen/Qwen3.5-9B": "Qwen9B",
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen7B",
    "openai/gpt-oss-120b": "Open120b",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "Deep0731",
    "google/gemma-4-31B-it": "gemma31B",
}


@dataclass(frozen=True)
class PromptTemplates:
    """Prompt components loaded from the local prompt configuration file."""

    command: str
    content: str
    template: str


def load_prompt_templates(prompt_path: Path) -> PromptTemplates:
    """Read the three string assignments in prompt.md without executing code."""
    try:
        source = prompt_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Unable to read prompt file {prompt_path}: {error}") from error

    try:
        tree = ast.parse(source, filename=str(prompt_path))
    except SyntaxError as error:
        raise ValueError(f"Invalid prompt file syntax in {prompt_path}: {error}") from error

    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in REQUIRED_PROMPT_NAMES:
                continue
            try:
                value = ast.literal_eval(value_node)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{target.id} must be a string literal") from error
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{target.id} must be a non-empty string")
            values[target.id] = value.strip()

    missing = [name for name in REQUIRED_PROMPT_NAMES if name not in values]
    if missing:
        raise ValueError(
            f"Prompt file {prompt_path} is missing: {', '.join(missing)}"
        )
    return PromptTemplates(
        command=values["COMMAND_PROMPT"],
        content=values["CONTENT_PROMPT"],
        template=values["TEMPLATE_PROMPT"],
    )


def load_raw_records(input_path: Path, problem_field: str) -> list[dict[str, Any]]:
    """Load and validate raw question records."""
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read input file {input_path}: {error}") from error

    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Blank line at {input_path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at {input_path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"Record at {input_path}:{line_number} must be an object")
        problem = record.get(problem_field)
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(
                f"Record at {input_path}:{line_number} needs a non-empty "
                f"{problem_field!r} field"
            )
        records.append(record)
    return records


def batch_record(
    *,
    custom_id: str,
    problem: str,
    parameters: dict[str, Any],
    prompts: PromptTemplates,
) -> dict[str, Any]:
    """Create one Together chat-completion Batch API record."""
    try:
        content = prompts.content.format(
            example_id=custom_id,
            problem_sentence=problem,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid placeholder in CONTENT_PROMPT: {error}") from error

    system_prompt = f"{prompts.command}\n\n{prompts.template}"
    body = {
        "model": parameters["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content.strip()},
        ],
        **{key: value for key, value in parameters.items() if key != "model"},
    }
    return {"custom_id": custom_id, "body": body}


def convert_records(
    records: Iterable[dict[str, Any]],
    *,
    problem_field: str,
    id_prefix: str,
    parameters: dict[str, Any],
    prompts: PromptTemplates,
) -> list[dict[str, Any]]:
    """Convert raw records while preserving their original line order in IDs."""
    return [
        batch_record(
            custom_id=f"{id_prefix}-{index:06d}",
            problem=record[problem_field],
            parameters=parameters,
            prompts=prompts,
        )
        for index, record in enumerate(records, start=1)
    ]


def model_short_name(model: str) -> str:
    """Return a configured label or a safe first-four/last-four fallback."""
    known_names = {
        model_id.casefold(): short_name
        for model_id, short_name in MODEL_SHORT_NAMES.items()
    }
    known_name = known_names.get(model.strip().casefold())
    if known_name:
        return known_name

    alphanumeric_model = "".join(character for character in model if character.isalnum())
    if not alphanumeric_model:
        raise ValueError("model must contain at least one letter or number")
    if len(alphanumeric_model) <= 8:
        return alphanumeric_model
    return f"{alphanumeric_model[:4]}{alphanumeric_model[-4:]}"


def default_output_path(input_path: Path, model: str) -> Path:
    """Build the fixed data/batch output path with the model short name."""
    filename = f"{input_path.stem}_{model_short_name(model)}_batch.jsonl"
    return BATCH_OUTPUT_DIRECTORY / filename


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write compact UTF-8 JSONL output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse converter paths plus the shared model hyperparameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/question_20_test.jsonl"),
        help="Raw JSONL input path (default: data/raw/question_20_test.jsonl)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help=f"Prompt configuration path (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument(
        "--problem-field",
        default="problem",
        help="Input field containing the model question (default: problem)",
    )
    parser.add_argument(
        "--id-prefix",
        default="question",
        help="Prefix for generated custom_id values (default: question)",
    )
    add_hyperparameter_arguments(parser)
    return parser.parse_args(argv)


def main() -> None:
    """Run the raw-to-Batch JSONL conversion pipeline."""
    args = parse_args()
    if not args.id_prefix or len(args.id_prefix) > 56:
        raise SystemExit("Conversion error: id-prefix must contain 1 to 56 characters")

    try:
        hyperparameters = hyperparameters_from_namespace(args)
        parameters = hyperparameters.to_request_parameters()
        output_path = default_output_path(args.input, parameters["model"])
        prompts = load_prompt_templates(args.prompt_file)
        raw_records = load_raw_records(args.input, args.problem_field)
        converted_records = convert_records(
            raw_records,
            problem_field=args.problem_field,
            id_prefix=args.id_prefix,
            parameters=parameters,
            prompts=prompts,
        )
        write_jsonl(converted_records, output_path)
    except ValueError as error:
        raise SystemExit(f"Conversion error: {error}") from error

    print(f"Converted {len(converted_records)} records")
    print(f"Input:  {args.input}")
    print(f"Output: {output_path}")
    print(f"Model:  {parameters['model']}")


if __name__ == "__main__":
    main()
