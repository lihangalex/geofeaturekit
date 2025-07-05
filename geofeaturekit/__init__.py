"""GeoFeatureKit - Urban feature extraction and analysis toolkit."""

from typing import Union, Dict, List, Tuple
from .core.extractor import UrbanFeatureExtractor
from .exceptions.errors import GeoFeatureKitError

def features_from_location(
    location: Union[Dict[str, Union[float, int]], Tuple[float, float, int]],
    show_progress: bool = True,
    progress_detail: str = 'normal',
    cache: bool = True
) -> Dict[str, any]:
    """Extract urban features from a location.
    
    Args:
        location: Either a dictionary with 'latitude', 'longitude', and 'radius_meters' keys,
                 or a tuple of (latitude, longitude, radius_meters)
        show_progress: Whether to show progress information
        progress_detail: Level of progress detail ('minimal', 'normal', 'verbose')
        cache: Whether to use caching for downloaded data
        
    Returns:
        Dictionary containing urban features and metrics
        
    Raises:
        ValueError: If location format is invalid
        GeoFeatureKitError: If feature extraction fails
    """
    # Parse location input
    if isinstance(location, dict):
        latitude = location['latitude']
        longitude = location['longitude']
        radius_meters = location['radius_meters']
    elif isinstance(location, tuple) and len(location) == 3:
        latitude, longitude, radius_meters = location
    else:
        raise ValueError("Location must be either dict with lat/lon/radius keys or (lat, lon, radius) tuple")
    
    # Validate radius
    if radius_meters <= 0:
        raise ValueError("Radius must be positive")
    
    # Warn about very small radii
    if radius_meters < 50:
        print(f"Warning: Very small radius ({radius_meters}m) may result in limited or no data.")
        print("Consider using a radius of at least 50m for meaningful analysis.")
    elif radius_meters < 100:
        print(f"Warning: Small radius ({radius_meters}m) may result in limited data.")
        print("Consider using a radius of at least 100m for comprehensive analysis.")
    
    try:
        # Create extractor
        extractor = UrbanFeatureExtractor(
            radius_meters=radius_meters,
            use_cache=cache,
            show_progress=show_progress,
            progress_detail=progress_detail
        )
        
        # Extract features
        return extractor.features_from_location(
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius_meters
        )
        
    except Exception as e:
        raise GeoFeatureKitError(f"Error extracting features: {str(e)}") 