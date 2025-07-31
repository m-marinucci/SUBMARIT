"""
SUBMARIT - SUBMARket Identification and Testing

A Python implementation of submarket clustering algorithms for analyzing
product substitution patterns.
"""

__version__ = "0.1.0"
__author__ = "SUBMARIT Contributors"

from .algorithms import (
    KSMLocalSearch,
    KSMLocalSearch2,
    KSMLocalSearchConstrained,
    KSMLocalSearchConstrained2,
    LocalSearchResult,
)

__all__ = [
    "KSMLocalSearch",
    "KSMLocalSearch2",
    "KSMLocalSearchConstrained",
    "KSMLocalSearchConstrained2",
    "LocalSearchResult",
]