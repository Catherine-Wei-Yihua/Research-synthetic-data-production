"""Random sampling of coefficients, monomials, and polynomial terms."""

from __future__ import annotations

import random
from fractions import Fraction

from .config import (
    COEFFICIENT_MAX,
    COEFFICIENT_MIN,
    FRACTION_DENOMINATOR_MAX,
    FULL_RANGE_PROBABILITY,
    INTEGER_PROBABILITY,
    SMALL_INTEGER_MAX,
    SMALL_INTEGER_MIN,
)
from .formatting import render_terms
from .models import Monomial, Term


def sample_nonzero_coefficient(rng: random.Random) -> Fraction:
    for _ in range(10_000):
        if rng.random() < INTEGER_PROBABILITY:
            if rng.random() < FULL_RANGE_PROBABILITY:
                value = rng.randint(COEFFICIENT_MIN, COEFFICIENT_MAX)
            else:
                value = rng.randint(SMALL_INTEGER_MIN, SMALL_INTEGER_MAX)
            if value != 0:
                return Fraction(value)
        else:
            numerator = rng.randint(COEFFICIENT_MIN, COEFFICIENT_MAX)
            denominator = rng.randint(2, FRACTION_DENOMINATOR_MAX)
            value = Fraction(numerator, denominator)
            if value != 0 and value.denominator != 1:
                return value
    raise RuntimeError("Unable to sample a nonzero coefficient")


def all_monomials(
    variable_count: int,
    max_degree: int,
    include_constant: bool = True,
) -> list[Monomial]:
    result: list[Monomial] = []
    for total_degree in range(0 if include_constant else 1, max_degree + 1):
        if variable_count == 1:
            result.append((total_degree, 0))
        else:
            for x_power in range(total_degree, -1, -1):
                result.append((x_power, total_degree - x_power))
    return result


def choose_monomials(
    rng: random.Random,
    *,
    variable_count: int,
    max_degree: int,
    count: int,
    include_constant: bool,
    require_both_variables: bool = False,
) -> list[Monomial]:
    pool = all_monomials(variable_count, max_degree, include_constant=True)
    if not include_constant:
        pool = [monomial for monomial in pool if monomial != (0, 0)]
    if count > len(pool):
        raise ValueError("Requested more unique monomials than are available")

    for _ in range(10_000):
        chosen = rng.sample(pool, count)
        if include_constant and (0, 0) not in chosen:
            continue
        if not any(monomial != (0, 0) for monomial in chosen):
            continue
        if require_both_variables:
            has_x = any(monomial[0] > 0 for monomial in chosen)
            has_y = any(monomial[1] > 0 for monomial in chosen)
            if not (has_x and has_y):
                continue
        return chosen
    raise RuntimeError("Unable to choose a valid monomial set")


def random_polynomial_terms(
    rng: random.Random,
    *,
    variable_count: int,
    max_degree: int,
    term_count: int,
    include_constant: bool = True,
    require_both_variables: bool = False,
) -> list[Term]:
    monomials = choose_monomials(
        rng,
        variable_count=variable_count,
        max_degree=max_degree,
        count=term_count,
        include_constant=include_constant,
        require_both_variables=require_both_variables,
    )
    return [(sample_nonzero_coefficient(rng), monomial) for monomial in monomials]


def shuffled_render(rng: random.Random, terms: list[Term]) -> str:
    terms = list(terms)
    rng.shuffle(terms)
    return render_terms(terms)

