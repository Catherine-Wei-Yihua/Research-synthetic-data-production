"""Rendering helpers for polynomial expressions."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

from .models import Monomial, Polynomial, Term
from .polynomial import clean_poly


def fraction_text(value: Fraction) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def monomial_text(monomial: Monomial) -> str:
    x_power, y_power = monomial
    factors: list[str] = []
    if x_power:
        factors.append("x" if x_power == 1 else f"x^{x_power}")
    if y_power:
        factors.append("y" if y_power == 1 else f"y^{y_power}")
    return "*".join(factors)


def unsigned_term_text(coefficient: Fraction, monomial: Monomial) -> str:
    coefficient = abs(coefficient)
    variable = monomial_text(monomial)
    if not variable:
        return fraction_text(coefficient)
    if coefficient == 1:
        return variable
    return f"{fraction_text(coefficient)}*{variable}"


def render_terms(terms: Iterable[Term]) -> str:
    pieces: list[str] = []
    for coefficient, monomial in terms:
        if coefficient == 0:
            continue
        body = unsigned_term_text(coefficient, monomial)
        if not pieces:
            pieces.append(f"-{body}" if coefficient < 0 else body)
        else:
            pieces.append((" - " if coefficient < 0 else " + ") + body)
    return "".join(pieces) if pieces else "0"


def monomial_sort_key(monomial: Monomial) -> tuple[int, int, int]:
    x_power, y_power = monomial
    return -(x_power + y_power), -x_power, -y_power


def render_poly(poly: Polynomial) -> str:
    ordered = sorted(clean_poly(poly).items(), key=lambda item: monomial_sort_key(item[0]))
    return render_terms((coefficient, monomial) for monomial, coefficient in ordered)


def scalar_times_parentheses_text(scalar: Fraction, inside: str) -> str:
    magnitude = abs(scalar)
    if magnitude == 1:
        return f"({inside})"
    return f"{fraction_text(magnitude)}*({inside})"


def monomial_factor_text(coefficient: Fraction, monomial: Monomial) -> str:
    return unsigned_term_text(abs(coefficient), monomial)


def join_signed_pieces(pieces: Iterable[tuple[int, str]]) -> str:
    output: list[str] = []
    for sign, body in pieces:
        if not output:
            output.append(f"-{body}" if sign < 0 else body)
        else:
            output.append((" - " if sign < 0 else " + ") + body)
    return "".join(output) if output else "0"


def parenthesis_depth(text: str) -> int:
    current = 0
    maximum = 0
    for character in text:
        if character == "(":
            current += 1
            maximum = max(maximum, current)
        elif character == ")":
            current -= 1
            if current < 0:
                raise ValueError(f"Unbalanced parentheses: {text}")
    if current != 0:
        raise ValueError(f"Unbalanced parentheses: {text}")
    return maximum


def parenthesis_pair_count(text: str) -> int:
    # Validate that the parentheses are balanced before counting their pairs.
    parenthesis_depth(text)
    return text.count("(")

