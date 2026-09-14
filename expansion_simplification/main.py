#!/usr/bin/env python3
"""Generate middle-school Expansion and Simplification problems.

Four categories are generated in equal proportions:
  1. combine_like_terms
  2. monomial_times_polynomial
  3. nested_parentheses
  4. polynomial_times_polynomial

The output is JSONL. By default each record contains only ``problem`` and
``answer``. Pass ``--include-category`` to add the category label.

Only Python's standard library is required. Coefficients in the generated
problems are integers or reduced fractions; decimals are never used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from raw_question_generation.dataset import generate_dataset, print_summary, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the generation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Number of questions (default: 100)")
    parser.add_argument("--seed", type=int, default=202608162, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; default: expansion_simplification_<n>.jsonl beside this script",
    )
    parser.add_argument(
        "--include-category",
        action="store_true",
        help="Include the category field in each JSONL record",
    )
    return parser.parse_args()


def main() -> None:
    """Run generation, serialization, and summary reporting."""
    args = parse_args()
    output_path = args.output or Path(__file__).with_name(
        f"expansion_simplification_{args.n}.jsonl"
    )

    # Build and validate the complete, category-balanced dataset.
    records, candidates = generate_dataset(
        args.n,
        args.seed,
        include_category=args.include_category,
    )

    # Persist the records and report the category distribution.
    write_jsonl(records, output_path)
    print_summary(candidates, output_path)


if __name__ == "__main__":
    main()
