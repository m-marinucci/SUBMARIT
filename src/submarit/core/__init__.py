"""Core functionality for SUBMARIT."""

from .create_substitution_matrix import (
    create_substitution_matrix,
    create_substitution_matrix_from_data,
)
from .substitution_matrix import SubstitutionMatrix

__all__ = [
    "SubstitutionMatrix",
    "create_substitution_matrix",
    "create_substitution_matrix_from_data",
]