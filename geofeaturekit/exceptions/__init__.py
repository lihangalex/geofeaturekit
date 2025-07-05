"""Custom exceptions for GeoFeatureKit."""

class GeoFeatureKitError(Exception):
    """Base exception class for GeoFeatureKit errors."""
    pass 

"""Exceptions package initialization."""

from .errors import GeoFeatureKitError 