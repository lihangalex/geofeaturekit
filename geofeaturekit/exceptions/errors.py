"""Custom exceptions for the GeoFeatureKit package."""

class GeoFeatureKitError(Exception):
    """Base exception for all GeoFeatureKit errors."""
    pass

class LocationError(GeoFeatureKitError):
    """Error related to location data or geocoding."""
    pass

class NetworkError(GeoFeatureKitError):
    """Error related to network operations or analysis."""
    pass

class POIError(GeoFeatureKitError):
    """Error related to POI data or analysis."""
    pass

class RateLimitError(GeoFeatureKitError):
    """Error related to API rate limiting."""
    pass

class ConfigurationError(GeoFeatureKitError):
    """Error related to configuration settings."""
    pass

class OutputError(GeoFeatureKitError):
    """Error related to output operations."""
    pass 