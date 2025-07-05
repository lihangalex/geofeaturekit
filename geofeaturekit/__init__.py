"""GeoFeatureKit - Urban feature extraction and analysis toolkit."""

from typing import Union, Dict, List
from .core.extractor import UrbanFeatureExtractor
from .exceptions.errors import GeoFeatureKitError

def features_from_location(
    locations: Union[Dict[str, float], List[Dict[str, float]]],
    use_cache: bool = True,
    show_progress: bool = True,
    progress_detail: str = 'normal'
) -> Union[Dict[str, any], Dict[str, Dict[str, any]]]:
    """Extract features from one or more locations.
    
    This is the main entry point for GeoFeatureKit. It provides a simple way
    to extract features from locations.
    
    Args:
        locations: Either a single location dictionary or a list of location dictionaries.
                  Each location should have 'latitude', 'longitude', and 'radius_meters'.
        use_cache: Whether to cache downloaded data (default: True)
        show_progress: Whether to show progress bars (default: True)
        progress_detail: Level of progress detail - 'minimal', 'normal', or 'verbose' (default: 'normal')
    
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
        
        >>> # With custom progress settings
        >>> features = features_from_location(
        ...     {"latitude": 40.7580, "longitude": -73.9855, "radius_meters": 500},
        ...     show_progress=False  # Silent mode
        ... )
    """
    try:
        extractor = UrbanFeatureExtractor(
            use_cache=use_cache,
            show_progress=show_progress,
            progress_detail=progress_detail
        )
        
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
            # Multiple locations - use batch processing
            try:
                location_tuples = []
                location_radii = {}
                
                for i, loc in enumerate(locations):
                    if not isinstance(loc, dict):
                        raise GeoFeatureKitError(f"Location {i} is not a dictionary")
                    
                    lat = loc["latitude"]
                    lon = loc["longitude"]
                    radius = loc["radius_meters"]
                    
                    location_tuples.append((lat, lon))
                    location_radii[f"{lat},{lon}"] = radius
                
                # For batch processing, we need to handle different radii
                # For now, we'll process them individually but with batch progress
                results = {}
                
                for i, (lat, lon) in enumerate(location_tuples):
                    key = f"{lat},{lon}"
                    radius = location_radii[key]
                    
                    try:
                        result = extractor.features_from_location(
                            latitude=lat,
                            longitude=lon,
                            radius_meters=radius
                        )
                        results[key] = result
                    except Exception as e:
                        results[f"location_{i}"] = {
                            "error": f"Error extracting features: {str(e)}"
                        }
                
                return results
                
            except KeyError as e:
                raise GeoFeatureKitError(f"Missing required field in location: {str(e)}")
            except Exception as e:
                raise GeoFeatureKitError(f"Error processing locations: {str(e)}")
            
    except Exception as e:
        raise GeoFeatureKitError(f"Error initializing feature extractor: {str(e)}") 