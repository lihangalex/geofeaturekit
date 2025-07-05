"""GeoFeatureKit - Urban feature extraction and analysis toolkit."""

from typing import Union, Dict, List
from .core.extractor import UrbanFeatureExtractor
from .exceptions.errors import GeoFeatureKitError

def features_from_location(
    locations: Union[Dict[str, float], List[Dict[str, float]]],
    use_cache: bool = True
) -> Union[Dict[str, any], Dict[str, Dict[str, any]]]:
    """Extract features from one or more locations.
    
    This is the main entry point for GeoFeatureKit. It provides a simple way
    to extract features from locations.
    
    Args:
        locations: Either a single location dictionary or a list of location dictionaries.
                  Each location should have 'latitude', 'longitude', and 'radius_meters'.
        use_cache: Whether to cache downloaded data (default: True)
    
    Returns:
        For a single location: Dictionary containing extracted features
        For multiple locations: Dictionary mapping location strings to extracted features
        
    Raises:
        GeoFeatureKitError: If there's an error during feature extraction
    
    Example:
        >>> # Single location
        >>> features = features_from_location({
        ...     "latitude": 40.7580,
        ...     "longitude": -73.9855,
        ...     "radius_meters": 500
        ... })
        
        >>> # Multiple locations
        >>> features = features_from_location([
        ...     {
        ...         "latitude": 40.7580,
        ...         "longitude": -73.9855,
        ...         "radius_meters": 500
        ...     },
        ...     {
        ...         "latitude": 40.7829,
        ...         "longitude": -73.9654,
        ...         "radius_meters": 300
        ...     }
        ... ])
    """
    try:
        extractor = UrbanFeatureExtractor(use_cache=use_cache)
        
        if isinstance(locations, dict):
            # Single location
            try:
                return extractor.features_from_location(
                    latitude=locations["latitude"],
                    longitude=locations["longitude"],
                    radius_meters=locations["radius_meters"]
                )
            except KeyError as e:
                raise GeoFeatureKitError(f"Missing required field in location: {str(e)}")
            except Exception as e:
                raise GeoFeatureKitError(f"Error extracting features: {str(e)}")
        else:
            # Multiple locations
            results = {}
            for i, loc in enumerate(locations):
                try:
                    if not isinstance(loc, dict):
                        raise GeoFeatureKitError(f"Location {i} is not a dictionary")
                    
                    results[f"{loc['latitude']},{loc['longitude']}"] = extractor.features_from_location(
                        latitude=loc["latitude"],
                        longitude=loc["longitude"],
                        radius_meters=loc["radius_meters"]
                    )
                except KeyError as e:
                    results[f"location_{i}"] = {
                        "error": f"Missing required field: {str(e)}"
                    }
                except Exception as e:
                    results[f"location_{i}"] = {
                        "error": f"Error extracting features: {str(e)}"
                    }
            return results
            
    except Exception as e:
        raise GeoFeatureKitError(f"Error initializing feature extractor: {str(e)}") 