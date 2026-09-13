"""Full-dataset generation, validation, serialization, and reporting."""

from __future__ import annotations

import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from .config import (
    CATEGORY_ORDER,
    COEFFICIENT_MAX,
    COEFFICIENT_MIN,
    FRACTION_DENOMINATOR_MAX,
    MAX_PARENTHESIS_PAIRS,
    NESTED_PARENTHESES_MAX_DEGREE,
)
from .formatting import parenthesis_depth, parenthesis_pair_count, render_poly
from .generators import GENERATORS
from .models import Candidate
from .polynomial import result_coefficients_are_allowed


def balanced_counts(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(CATEGORY_ORDER))
    return {
        category: base + (index < remainder)
        for index, category in enumerate(CATEGORY_ORDER)
    }


def validate_source_coefficient(coefficient: Fraction) -> None:
    if not Fraction(COEFFICIENT_MIN) <= coefficient <= Fraction(COEFFICIENT_MAX):
        raise AssertionError(f"Source coefficient outside [-99, 99]: {coefficient}")
    if coefficient.denominator > 1:
        if abs(coefficient.numerator) > 99 or coefficient.denominator > FRACTION_DENOMINATOR_MAX:
            raise AssertionError(f"Invalid source fraction: {coefficient}")


def validate_candidate(candidate: Candidate) -> None:
    if not candidate.answer_poly:
        raise AssertionError("A generated answer cannot be the zero polynomial")
    if not result_coefficients_are_allowed(candidate.answer_poly):
        raise AssertionError("A result coefficient is outside [-99, 99]")
    if "." in candidate.problem or "." in render_poly(candidate.answer_poly):
        raise AssertionError("Decimal notation is not allowed")
    for coefficient in candidate.source_coefficients:
        validate_source_coefficient(coefficient)

    depth = parenthesis_depth(candidate.problem)
    pair_count = parenthesis_pair_count(candidate.problem)
    if pair_count > MAX_PARENTHESIS_PAIRS:
        raise AssertionError(
            f"Question has {pair_count} parenthesis pairs; maximum is "
            f"{MAX_PARENTHESIS_PAIRS}"
        )
    if candidate.category == "combine_like_terms" and depth != 0:
        raise AssertionError("Combine-like-terms questions cannot contain parentheses")
    if (
        candidate.category in ("monomial_times_polynomial", "polynomial_times_polynomial")
        and depth != 1
    ):
        raise AssertionError(f"Unexpected parenthesis depth for {candidate.category}: {depth}")
    if candidate.category == "nested_parentheses" and depth != 2:
        raise AssertionError("Nested-parentheses questions must have exactly two levels")
    if candidate.category == "nested_parentheses" and any(
        sum(monomial) > NESTED_PARENTHESES_MAX_DEGREE
        for monomial in candidate.answer_poly
    ):
        raise AssertionError(
            "Nested-parentheses questions cannot exceed total degree "
            f"{NESTED_PARENTHESES_MAX_DEGREE}"
        )


def generate_dataset(
    n: int,
    seed: int,
    *,
    include_category: bool = False,
) -> tuple[list[dict[str, str]], list[Candidate]]:
    if n < 4:
        raise ValueError("n must be at least 4 so that all four categories are represented")

    rng = random.Random(seed)
    quotas = balanced_counts(n)
    candidates: list[Candidate] = []
    seen_problems: set[str] = set()

    for category in CATEGORY_ORDER:
        generator = GENERATORS[category]
        for _ in range(quotas[category]):
            for _ in range(30_000):
                candidate = generator(rng)
                if candidate.problem in seen_problems:
                    continue
                validate_candidate(candidate)
                seen_problems.add(candidate.problem)
                candidates.append(candidate)
                break
            else:
                raise RuntimeError(f"Unable to generate a unique {category} problem")

    rng.shuffle(candidates)
    records: list[dict[str, str]] = []
    for candidate in candidates:
        record = {
            "problem": f"Simplify the polynomial equation ${candidate.problem}$.",
            "correct answer": render_poly(candidate.answer_poly),
        }
        if include_category:
            record["category"] = candidate.category
        records.append(record)
    return records, candidates


def write_jsonl(records: Iterable[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def print_summary(candidates: list[Candidate], output_path: Path) -> None:
    counts = Counter(candidate.category for candidate in candidates)
    print(f"Wrote {len(candidates)} questions to {output_path}")
    for category in CATEGORY_ORDER:
        count = counts[category]
        print(f"  {category}: {count} ({count / len(candidates):.1%})")
