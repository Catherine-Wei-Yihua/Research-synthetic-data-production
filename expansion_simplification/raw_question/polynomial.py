"""Exact polynomial arithmetic."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

from .config import COEFFICIENT_MAX, COEFFICIENT_MIN, ENFORCE_RESULT_COEFFICIENT_RANGE
from .models import Monomial, Polynomial, Term


def clean_poly(poly: Polynomial) -> Polynomial:
    return {monomial: coefficient for monomial, coefficient in poly.items() if coefficient != 0}


def poly_from_terms(terms: Iterable[Term]) -> Polynomial:
    result: Polynomial = {}
    for coefficient, monomial in terms:
        result[monomial] = result.get(monomial, Fraction(0)) + coefficient
    return clean_poly(result)


def poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, Fraction(0)) + coefficient
    return clean_poly(result)


def poly_scale(poly: Polynomial, scalar: Fraction) -> Polynomial:
    return clean_poly({monomial: scalar * coefficient for monomial, coefficient in poly.items()})


def monomial_multiply(left: Monomial, right: Monomial) -> Monomial:
    return left[0] + right[0], left[1] + right[1]


def poly_mul(left: Polynomial, right: Polynomial) -> Polynomial:
    terms: list[Term] = []
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            terms.append(
                (
                    left_coefficient * right_coefficient,
                    monomial_multiply(left_monomial, right_monomial),
                )
            )
    return poly_from_terms(terms)


def multiply_by_monomial(
    poly: Polynomial,
    coefficient: Fraction,
    monomial: Monomial,
) -> Polynomial:
    return clean_poly(
        {
            monomial_multiply(term_monomial, monomial): coefficient * term_coefficient
            for term_monomial, term_coefficient in poly.items()
        }
    )


def result_coefficients_are_allowed(poly: Polynomial) -> bool:
    if not ENFORCE_RESULT_COEFFICIENT_RANGE:
        return True
    return all(
        Fraction(COEFFICIENT_MIN) <= coefficient <= Fraction(COEFFICIENT_MAX)
        for coefficient in poly.values()
    )

