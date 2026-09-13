"""Configuration constants for expansion and simplification generation."""

COEFFICIENT_MIN = -99
COEFFICIENT_MAX = 99

# A denominator cap of 12 keeps the default output at a middle-school level.
# All coefficients are integers in the current configuration.
FRACTION_DENOMINATOR_MAX = None
INTEGER_PROBABILITY = 1.0

# Most integer coefficients are kept fairly small for readable middle-school
# exercises, while some are sampled from the full requested [-99, 99] range.
SMALL_INTEGER_MIN = -12
SMALL_INTEGER_MAX = 12
FULL_RANGE_PROBABILITY = 0.20

# Keep the coefficients of the fully simplified answer in the same numeric
# range as the source expression. Set this to False if larger expanded results
# are acceptable for a future dataset.
ENFORCE_RESULT_COEFFICIENT_RANGE = True

NESTED_PARENTHESES_MAX_DEGREE = 2
MAX_PARENTHESIS_PAIRS = 3

CATEGORY_ORDER = (
    "combine_like_terms",
    "monomial_times_polynomial",
    "nested_parentheses",
    "polynomial_times_polynomial",
)

