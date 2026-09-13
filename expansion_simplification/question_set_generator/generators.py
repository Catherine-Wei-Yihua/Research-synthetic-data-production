"""Candidate generators for the four problem categories."""

from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction
from typing import Callable

from .config import MAX_PARENTHESIS_PAIRS, NESTED_PARENTHESES_MAX_DEGREE
from .formatting import (
    join_signed_pieces,
    monomial_factor_text,
    parenthesis_depth,
    scalar_times_parentheses_text,
)
from .models import Candidate, Monomial, Polynomial, Term
from .polynomial import (
    multiply_by_monomial,
    poly_add,
    poly_from_terms,
    poly_mul,
    poly_scale,
    result_coefficients_are_allowed,
)
from .sampling import (
    all_monomials,
    choose_monomials,
    random_polynomial_terms,
    sample_nonzero_coefficient,
    shuffled_render,
)


def generate_combine_like_terms(rng: random.Random) -> Candidate:
    for _ in range(20_000):
        variable_count = 2 if rng.random() < 0.40 else 1
        max_degree = rng.randint(1, 3)
        available = len(all_monomials(variable_count, max_degree, include_constant=True))
        minimum_unique = 3 if variable_count == 2 else 2
        unique_count = rng.randint(minimum_unique, min(5, available))
        monomials = choose_monomials(
            rng,
            variable_count=variable_count,
            max_degree=max_degree,
            count=unique_count,
            include_constant=True,
            require_both_variables=variable_count == 2,
        )

        raw_terms: list[Term] = []
        coefficients: list[Fraction] = []
        for monomial in monomials:
            repetitions = rng.choice((2, 2, 3))
            for _ in range(repetitions):
                coefficient = sample_nonzero_coefficient(rng)
                coefficients.append(coefficient)
                raw_terms.append((coefficient, monomial))

        answer = poly_from_terms(raw_terms)
        if (
            not answer
            or len(answer) < 2
            or len(answer) >= len(raw_terms)
            or not result_coefficients_are_allowed(answer)
        ):
            continue
        problem = shuffled_render(rng, raw_terms)
        return Candidate(problem, answer, "combine_like_terms", tuple(coefficients))
    raise RuntimeError("Unable to generate a combine-like-terms problem")


def generate_monomial_times_polynomial(rng: random.Random) -> Candidate:
    for _ in range(30_000):
        variable_count = 2 if rng.random() < 0.30 else 1
        group_count = rng.randint(2, MAX_PARENTHESIS_PAIRS)
        groups: list[tuple[int, str]] = []
        group_polys: list[Polynomial] = []
        source_coefficients: list[Fraction] = []
        contribution_counts: Counter[Monomial] = Counter()

        for _ in range(group_count):
            outer_degree = rng.randint(1, 2)
            outer_candidates = [
                monomial
                for monomial in all_monomials(variable_count, outer_degree, include_constant=False)
                if sum(monomial) == outer_degree
            ]
            outer_monomial = rng.choice(outer_candidates)
            outer_coefficient = sample_nonzero_coefficient(rng)
            inside_max_degree = max(1, 3 - outer_degree)
            inside_count = rng.randint(
                2,
                min(4, len(all_monomials(variable_count, inside_max_degree))),
            )
            inside_terms = random_polynomial_terms(
                rng,
                variable_count=variable_count,
                max_degree=inside_max_degree,
                term_count=inside_count,
                include_constant=True,
                require_both_variables=False,
            )
            inside_poly = poly_from_terms(inside_terms)
            contribution = multiply_by_monomial(inside_poly, outer_coefficient, outer_monomial)
            group_polys.append(contribution)
            contribution_counts.update(contribution.keys())

            factor = monomial_factor_text(outer_coefficient, outer_monomial)
            inside_text = shuffled_render(rng, inside_terms)
            groups.append((-1 if outer_coefficient < 0 else 1, f"{factor}*({inside_text})"))
            source_coefficients.extend(
                [outer_coefficient, *(coefficient for coefficient, _ in inside_terms)]
            )

        # At least one expanded monomial must occur in multiple groups, so the
        # problem requires both expansion and combining like terms.
        if not any(count >= 2 for count in contribution_counts.values()):
            continue
        answer: Polynomial = {}
        for group_poly in group_polys:
            answer = poly_add(answer, group_poly)
        if not answer or not result_coefficients_are_allowed(answer):
            continue
        problem = join_signed_pieces(groups)
        return Candidate(
            problem,
            answer,
            "monomial_times_polynomial",
            tuple(source_coefficients),
        )
    raise RuntimeError("Unable to generate a monomial-times-polynomial problem")


def generate_nested_parentheses(rng: random.Random) -> Candidate:
    for _ in range(30_000):
        variable_count = 2 if rng.random() < 0.25 else 1
        # Each top-level group contains one outer and one inner pair. Keeping a
        # single group ensures the equation stays below the global three-pair cap.
        top_group_count = 1
        top_pieces: list[tuple[int, str]] = []
        answer: Polynomial = {}
        source_coefficients: list[Fraction] = []

        for _ in range(top_group_count):
            outer_scalar = sample_nonzero_coefficient(rng)
            inner_scalar = sample_nonzero_coefficient(rng)
            base_count = rng.randint(2, 3)
            inner_count = rng.randint(2, 3)

            base_terms = random_polynomial_terms(
                rng,
                variable_count=variable_count,
                max_degree=NESTED_PARENTHESES_MAX_DEGREE,
                term_count=base_count,
                include_constant=True,
            )
            base_monomials = [monomial for _, monomial in base_terms]
            # Force at least one like term across the inner and outer content.
            shared_monomial = rng.choice(base_monomials)
            inner_pool = all_monomials(
                variable_count,
                NESTED_PARENTHESES_MAX_DEGREE,
                include_constant=True,
            )
            other_pool = [monomial for monomial in inner_pool if monomial != shared_monomial]
            inner_monomials = [shared_monomial, *rng.sample(other_pool, inner_count - 1)]
            inner_terms = [
                (sample_nonzero_coefficient(rng), monomial) for monomial in inner_monomials
            ]

            base_poly = poly_from_terms(base_terms)
            inner_poly = poly_from_terms(inner_terms)
            content_poly = poly_add(base_poly, poly_scale(inner_poly, inner_scalar))
            group_poly = poly_scale(content_poly, outer_scalar)
            answer = poly_add(answer, group_poly)

            base_text = shuffled_render(rng, base_terms)
            inner_text = shuffled_render(rng, inner_terms)
            inner_body = scalar_times_parentheses_text(inner_scalar, inner_text)
            content_text = join_signed_pieces(
                (
                    (1, base_text),
                    (-1 if inner_scalar < 0 else 1, inner_body),
                )
            )
            outer_body = scalar_times_parentheses_text(outer_scalar, content_text)
            top_pieces.append((-1 if outer_scalar < 0 else 1, outer_body))
            source_coefficients.extend(
                [
                    outer_scalar,
                    inner_scalar,
                    *(coefficient for coefficient, _ in base_terms),
                    *(coefficient for coefficient, _ in inner_terms),
                ]
            )

        if not answer or not result_coefficients_are_allowed(answer):
            continue
        problem = join_signed_pieces(top_pieces)
        if parenthesis_depth(problem) != 2:
            continue
        return Candidate(problem, answer, "nested_parentheses", tuple(source_coefficients))
    raise RuntimeError("Unable to generate a nested-parentheses problem")


def random_factor(rng: random.Random, variable_count: int) -> tuple[list[Term], Polynomial]:
    term_count = rng.choice((2, 2, 3))
    terms = random_polynomial_terms(
        rng,
        variable_count=variable_count,
        max_degree=2,
        term_count=term_count,
        include_constant=True,
        require_both_variables=variable_count == 2 and term_count == 3,
    )
    return terms, poly_from_terms(terms)


def product_text(rng: random.Random, left_terms: list[Term], right_terms: list[Term]) -> str:
    return f"({shuffled_render(rng, left_terms)})*({shuffled_render(rng, right_terms)})"


def generate_polynomial_times_polynomial(rng: random.Random) -> Candidate:
    for _ in range(30_000):
        variable_count = 2 if rng.random() < 0.25 else 1
        mode = rng.choices(
            ("single", "product_plus_square", "square"),
            weights=(0.35, 0.40, 0.25),
        )[0]
        source_coefficients: list[Fraction] = []

        left_terms, left_poly = random_factor(rng, variable_count)
        source_coefficients.extend(coefficient for coefficient, _ in left_terms)

        if mode == "square":
            problem = f"({shuffled_render(rng, left_terms)})^2"
            answer = poly_mul(left_poly, left_poly)
        else:
            right_terms, right_poly = random_factor(rng, variable_count)
            source_coefficients.extend(coefficient for coefficient, _ in right_terms)
            first_text = product_text(rng, left_terms, right_terms)
            answer = poly_mul(left_poly, right_poly)

            if mode == "single":
                problem = first_text
            else:
                third_terms, third_poly = random_factor(rng, variable_count)
                source_coefficients.extend(coefficient for coefficient, _ in third_terms)
                second_text = f"({shuffled_render(rng, third_terms)})^2"
                sign = rng.choice((-1, 1))
                problem = join_signed_pieces(((1, first_text), (sign, second_text)))
                answer = poly_add(answer, poly_scale(poly_mul(third_poly, third_poly), sign))

        if not answer or not result_coefficients_are_allowed(answer):
            continue
        return Candidate(
            problem,
            answer,
            "polynomial_times_polynomial",
            tuple(source_coefficients),
        )
    raise RuntimeError("Unable to generate a polynomial-times-polynomial problem")


GENERATORS: dict[str, Callable[[random.Random], Candidate]] = {
    "combine_like_terms": generate_combine_like_terms,
    "monomial_times_polynomial": generate_monomial_times_polynomial,
    "nested_parentheses": generate_nested_parentheses,
    "polynomial_times_polynomial": generate_polynomial_times_polynomial,
}
