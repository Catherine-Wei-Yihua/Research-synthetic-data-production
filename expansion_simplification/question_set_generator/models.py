"""Shared type aliases and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias


# A monomial is represented by (power_of_x, power_of_y).
Monomial: TypeAlias = tuple[int, int]
Polynomial: TypeAlias = dict[Monomial, Fraction]
Term: TypeAlias = tuple[Fraction, Monomial]


@dataclass(frozen=True)
class Candidate:
    problem: str
    answer_poly: Polynomial
    category: str
    source_coefficients: tuple[Fraction, ...]

