"""GeoFeatureKit - Urban feature extraction and analysis toolkit."""

from typing import Union, Dict, List, Tuple, Optional
from .core.extractor import UrbanFeatureExtractor
from .exceptions.errors import GeoFeatureKitError
from .utils.isochrone import (
    extract_isochrone_features, 
    validate_speed_config, 
    get_default_speed_config
)

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


def features_from_coordinate(
    lat: float,
    lon: float,
    radius_m: Optional[float] = None,
    max_travel_time_min_walk: Optional[float] = None,
    max_travel_time_min_bike: Optional[float] = None,
    max_travel_time_min_drive: Optional[float] = None,
    speed_config: Optional[Dict[str, float]] = None,
    show_progress: bool = True,
    progress_detail: str = 'normal',
    cache: bool = True
) -> Dict[str, any]:
    """Extract urban features from coordinates with multi-modal accessibility analysis.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        radius_m: Optional radius in meters for radius-based analysis
        max_travel_time_min_walk: Optional maximum walking time in minutes
        max_travel_time_min_bike: Optional maximum biking time in minutes
        max_travel_time_min_drive: Optional maximum driving time in minutes
        speed_config: Optional speed configuration dict with 'walk', 'bike', 'drive' keys (km/h)
        show_progress: Whether to show progress information
        progress_detail: Level of progress detail ('minimal', 'normal', 'verbose')
        cache: Whether to use caching for downloaded data
        
    Returns:
        Dictionary containing features for each specified analysis type:
        - radius_features: Features within circular radius (if radius_m specified)
        - isochrone_features_walk: Features within walking isochrone (if max_travel_time_min_walk specified)
        - isochrone_features_bike: Features within biking isochrone (if max_travel_time_min_bike specified)
        - isochrone_features_drive: Features within driving isochrone (if max_travel_time_min_drive specified)
        
    Raises:
        ValueError: If coordinates are invalid or no analysis type specified
        TypeError: If speed values are not numeric
        GeoFeatureKitError: If feature extraction fails
        
    Example:
        >>> features = features_from_coordinate(
        ...     lat=40.7580, lon=-73.9855,
        ...     max_travel_time_min_walk=10,
        ...     max_travel_time_min_bike=5,
        ...     speed_config={'walk': 4.8, 'bike': 17}
        ... )
        >>> print(features.keys())
        dict_keys(['isochrone_features_walk', 'isochrone_features_bike'])
    """
    # Validate coordinates
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")
    
    # Check that at least one analysis type is specified
    analysis_types = [radius_m, max_travel_time_min_walk, max_travel_time_min_bike, max_travel_time_min_drive]
    if all(x is None for x in analysis_types):
        raise ValueError("At least one analysis type must be specified (radius_m or travel time parameters)")
    
    # Initialize speed configuration
    if speed_config is None:
        speed_config = get_default_speed_config()
    else:
        # Validate speed configuration
        validate_speed_config(speed_config)
    
    results = {}
    
    try:
        # Radius-based features
        if radius_m is not None:
            if radius_m <= 0:
                raise ValueError("Radius must be positive")
            
            extractor = UrbanFeatureExtractor(
                radius_meters=int(radius_m),
                use_cache=cache,
                show_progress=show_progress,
                progress_detail=progress_detail
            )
            
            results['radius_features'] = extractor.features_from_location(
                latitude=lat,
                longitude=lon,
                radius_meters=int(radius_m)
            )
        
        # Walking isochrone features
        if max_travel_time_min_walk is not None:
            if max_travel_time_min_walk <= 0:
                raise ValueError("Walk travel time must be positive")
            
            results['isochrone_features_walk'] = extract_isochrone_features(
                latitude=lat,
                longitude=lon,
                travel_time_minutes=max_travel_time_min_walk,
                mode='walk',
                speed_kmh=speed_config['walk'],
                show_progress=show_progress
            )
        
        # Biking isochrone features  
        if max_travel_time_min_bike is not None:
            if max_travel_time_min_bike <= 0:
                raise ValueError("Bike travel time must be positive")
            
            results['isochrone_features_bike'] = extract_isochrone_features(
                latitude=lat,
                longitude=lon,
                travel_time_minutes=max_travel_time_min_bike,
                mode='bike',
                speed_kmh=speed_config['bike'],
                show_progress=show_progress
            )
        
        # Driving isochrone features
        if max_travel_time_min_drive is not None:
            if max_travel_time_min_drive <= 0:
                raise ValueError("Drive travel time must be positive")
            
            results['isochrone_features_drive'] = extract_isochrone_features(
                latitude=lat,
                longitude=lon,
                travel_time_minutes=max_travel_time_min_drive,
                mode='drive',
                speed_kmh=speed_config['drive'],
                show_progress=show_progress
            )
        
        return results
        
    except Exception as e:
        raise GeoFeatureKitError(f"Error extracting coordinate features: {str(e)}") 