#!/usr/bin/env python3
"""Clean Together Batch responses and score polynomial final answers."""

from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "raw" / "expansion_5000_50002026.jsonl"
CLEAN_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "clean"

Monomial = tuple[int, int]
Polynomial = dict[Monomial, Fraction]


class AnswerParseError(ValueError):
    """Raised when a final answer is not a supported polynomial expression."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and validate that every line is a JSON object."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read {path}: {error}") from error
    if not lines:
        raise ValueError(f"JSONL file is empty: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Blank line at {path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Record at {path}:{line_number} must be a JSON object")
        records.append(record)
    return records


def parse_model_content(content: Any, line_number: int) -> dict[str, Any]:
    """Decode the JSON object stored in a Together response's content field."""
    if isinstance(content, dict):
        parsed = content
    elif isinstance(content, str):
        text = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Response line {line_number} contains invalid inner JSON: {error.msg}"
            ) from error
    else:
        raise ValueError(f"Response line {line_number} has non-string message content")

    if not isinstance(parsed, dict):
        raise ValueError(f"Response line {line_number} inner content must be a JSON object")
    return parsed


def extract_response_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    """Extract requested tidy fields from one nested Together Batch response."""
    try:
        status_code = record["response"]["status_code"]
        response_body = record["response"]["body"]
        message = response_body["choices"][0]["message"]
        content = parse_model_content(message["content"], line_number)
        tidy = {
            "id": record["id"],
            "model": response_body["model"],
            "question_id": record["custom_id"],
            "question": content["question"],
            "steps": content["steps"],
            "final_answer": content["final_answer"],
        }
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"Response line {line_number} is missing required data: {error}"
        ) from error

    if status_code != 200:
        raise ValueError(f"Response line {line_number} has HTTP status {status_code}")
    for field_name in ("id", "model", "question_id", "question", "final_answer"):
        value = tidy[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Response line {line_number} has invalid {field_name!r} content"
            )
    if not isinstance(tidy["steps"], list) or not all(
        isinstance(step, str) for step in tidy["steps"]
    ):
        raise ValueError(f"Response line {line_number} 'steps' must be a list of strings")
    return tidy


def normalize_question(question: str) -> str:
    """Normalize whitespace only for reference-question matching."""
    return "".join(unicodedata.normalize("NFKC", question).split())


def normalize_answer_text(answer: str) -> str:
    """Normalize harmless presentation differences without reordering terms."""
    normalized = unicodedata.normalize("NFKC", answer).strip()
    normalized = normalized.replace("−", "-").replace("–", "-")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = normalized.replace("×", "*").replace("·", "*")

    if normalized.startswith("$") and normalized.endswith("$"):
        normalized = normalized[1:-1].strip()
    boxed = re.fullmatch(r"\\boxed\{(.*)\}", normalized, flags=re.DOTALL)
    if boxed:
        normalized = boxed.group(1)

    normalized = re.sub(r"\^\{([+-]?\d+)\}", r"^\1", normalized)
    normalized = normalized.rstrip(".")
    normalized = "".join(normalized.split())
    normalized = normalized.replace("**", "^")
    return normalized


def clean_poly(poly: Polynomial) -> Polynomial:
    """Remove zero-coefficient terms from a polynomial."""
    return {monomial: coefficient for monomial, coefficient in poly.items() if coefficient}


def poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    """Add two exact polynomials."""
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, Fraction(0)) + coefficient
    return clean_poly(result)


def poly_scale(poly: Polynomial, scalar: Fraction) -> Polynomial:
    """Multiply a polynomial by an exact scalar."""
    return clean_poly({monomial: coefficient * scalar for monomial, coefficient in poly.items()})


def poly_mul(left: Polynomial, right: Polynomial) -> Polynomial:
    """Multiply two exact polynomials."""
    result: Polynomial = {}
    for (left_x, left_y), left_coefficient in left.items():
        for (right_x, right_y), right_coefficient in right.items():
            monomial = (left_x + right_x, left_y + right_y)
            result[monomial] = (
                result.get(monomial, Fraction(0))
                + left_coefficient * right_coefficient
            )
    return clean_poly(result)


def expression_for_parsing(answer: str) -> str:
    """Translate common plain-text/LaTeX polynomial notation into Python syntax."""
    expression = normalize_answer_text(answer)
    expression = re.sub(r"\\frac\{([+-]?\d+)\}\{(\d+)\}", r"(\1/\2)", expression)
    expression = expression.replace("^", "**")
    expression = re.sub(r"(?<=\d)(?=[xy(])", "*", expression)
    expression = re.sub(r"(?<=[xy)])(?=[xy(])", "*", expression)
    expression = re.sub(r"(?<=\))(?=\d)", "*", expression)
    return expression


def polynomial_from_ast(node: ast.AST) -> Polynomial:
    """Evaluate a restricted arithmetic AST as an exact x/y polynomial."""
    if isinstance(node, ast.Expression):
        return polynomial_from_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {(0, 0): Fraction(node.value)} if node.value else {}
    if isinstance(node, ast.Name) and node.id in {"x", "y"}:
        return {(1, 0) if node.id == "x" else (0, 1): Fraction(1)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = polynomial_from_ast(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else poly_scale(operand, Fraction(-1))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = polynomial_from_ast(node.left)
        right = polynomial_from_ast(node.right)
        signed_right = (
            right if isinstance(node.op, ast.Add) else poly_scale(right, Fraction(-1))
        )
        return poly_add(left, signed_right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return poly_mul(polynomial_from_ast(node.left), polynomial_from_ast(node.right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        numerator = polynomial_from_ast(node.left)
        denominator = polynomial_from_ast(node.right)
        if set(denominator) != {(0, 0)} or denominator[(0, 0)] == 0:
            raise AnswerParseError("division is only supported by a nonzero constant")
        return poly_scale(numerator, Fraction(1, 1) / denominator[(0, 0)])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
            raise AnswerParseError("powers must be integer literals")
        exponent = node.right.value
        if exponent < 0 or exponent > 20:
            raise AnswerParseError("power must be between 0 and 20")
        base = polynomial_from_ast(node.left)
        result: Polynomial = {(0, 0): Fraction(1)}
        for _ in range(exponent):
            result = poly_mul(result, base)
        return result
    raise AnswerParseError(f"unsupported expression element: {type(node).__name__}")


def parse_polynomial(answer: str) -> Polynomial:
    """Parse a supported polynomial answer into a canonical exact representation."""
    expression = expression_for_parsing(answer)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise AnswerParseError(f"invalid expression syntax: {answer!r}") from error
    return polynomial_from_ast(tree)


def answers_are_equal(model_answer: str, correct_answer: str) -> bool:
    """Compare answers mathematically, with normalized-text fallback."""
    try:
        return parse_polynomial(model_answer) == parse_polynomial(correct_answer)
    except AnswerParseError:
        return normalize_answer_text(model_answer).replace("*", "") == (
            normalize_answer_text(correct_answer).replace("*", "")
        )


def reference_answers(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Index reference answers by normalized question text."""
    references: dict[str, str] = {}
    for line_number, record in enumerate(records, start=1):
        question = record.get("problem")
        answer = record.get("correct answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Reference line {line_number} has no valid 'problem'")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Reference line {line_number} has no valid 'correct answer'")
        key = normalize_question(question)
        if key in references:
            raise ValueError(f"Duplicate reference question at line {line_number}")
        references[key] = answer
    return references


def clean_and_score(
    response_records: Iterable[dict[str, Any]],
    references: dict[str, str],
) -> list[dict[str, Any]]:
    """Flatten responses and add a boolean correctness result."""
    output: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for line_number, record in enumerate(response_records, start=1):
        tidy = extract_response_record(record, line_number)
        question_id = tidy["question_id"]
        if question_id in seen_question_ids:
            raise ValueError(
                f"Duplicate question_id {question_id!r} at response line {line_number}"
            )
        seen_question_ids.add(question_id)

        question_key = normalize_question(tidy["question"])
        if question_key not in references:
            raise ValueError(
                f"Response line {line_number} question_id {question_id!r} was not "
                "found in the reference dataset"
            )
        tidy["whether_correct"] = answers_are_equal(
            tidy["final_answer"],
            references[question_key],
        )
        output.append(tidy)
    return output


def default_output_path(input_path: Path) -> Path:
    """Return the default project-local path for cleaned responses."""
    return CLEAN_OUTPUT_DIRECTORY / f"{input_path.stem}_clean.jsonl"


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write tidy records as compact UTF-8 JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse response, reference, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Together Batch output JSONL")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help=f"Raw question reference JSONL (default: {DEFAULT_REFERENCE_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Clean JSONL path; default: data/clean/<input>_clean.jsonl",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run response cleaning and exact-answer scoring."""
    args = parse_args()
    output_path = args.output or default_output_path(args.input)
    if output_path.resolve() in {args.input.resolve(), args.reference.resolve()}:
        raise SystemExit("Cleaning error: output cannot overwrite an input file")

    try:
        response_records = read_jsonl(args.input)
        reference_records = read_jsonl(args.reference)
        references = reference_answers(reference_records)
        cleaned_records = clean_and_score(response_records, references)
        write_jsonl(cleaned_records, output_path)
    except ValueError as error:
        raise SystemExit(f"Cleaning error: {error}") from error

    correct_count = sum(record["whether_correct"] for record in cleaned_records)
    print(f"Cleaned {len(cleaned_records)} records")
    print(f"Correct:   {correct_count}")
    print(f"Incorrect: {len(cleaned_records) - correct_count}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
