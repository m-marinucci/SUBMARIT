"""
SUBMARIT - SUBMARket Identification and Testing

A Python implementation of submarket clustering algorithms for analyzing
product substitution patterns.
"""

__version__ = "0.1.0"
__author__ = "Numinate Consulting"
__email__ = "info@numinate.com"
__url__ = "www.numinate.com"

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