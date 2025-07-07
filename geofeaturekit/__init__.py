"""GeoFeatureKit - Urban feature extraction and analysis toolkit."""

import logging
from typing import Dict, List, Tuple, Optional, Callable, Any

from .core.extractor import UrbanFeatureExtractor
from .exceptions.errors import GeoFeatureKitError
from .utils.isochrone import (
    extract_isochrone_features, 
    validate_speed_config, 
    get_default_speed_config
)

# Set up logger for this module
logger = logging.getLogger(__name__)

# Type definitions for better clarity
ProgressCallback = Callable[[str, float], None]


def extract_features(
    latitude: float,
    longitude: float,
    radius_meters: int,
    *,
    verbose: bool = False,
    cache: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Any]:
    """Extract urban features from a specific location.
    
    This is the core function for extracting geospatial features. It downloads
    street network and POI data, then computes comprehensive urban metrics.
    
    Args:
        latitude: Location latitude (-90 to 90)
        longitude: Location longitude (-180 to 180) 
        radius_meters: Analysis radius in meters (must be positive)
        verbose: Enable verbose output to console (default: False)
        cache: Whether to use caching for downloaded data (default: True)
        progress_callback: Optional callback function(message: str, progress: float)
                          where progress is 0.0 to 1.0
        
    Returns:
        Dictionary containing urban features and metrics with keys:
        - network_metrics: Street network connectivity and pattern analysis
        - poi_metrics: Point of interest density and distribution
        - units: Measurement units used
        
    Raises:
        ValueError: If coordinates are invalid or radius is non-positive
        GeoFeatureKitError: If feature extraction fails
        
    Example:
        >>> features = extract_features(40.7580, -73.9855, 500)
        >>> print(f"Total POIs: {features['poi_metrics']['absolute_counts']['total_points_of_interest']}")
        
        >>> # With progress callback
        >>> def progress_handler(message, progress):
        ...     print(f"[{progress:.0%}] {message}")
        >>> features = extract_features(40.7580, -73.9855, 500, progress_callback=progress_handler)
    """
    # Validate inputs
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90.")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180.")
    if radius_meters <= 0:
        raise ValueError(f"Radius must be positive, got: {radius_meters}")
    
    # Log warnings for small radii instead of printing
    if radius_meters < 50:
        logger.warning(
            f"Very small radius ({radius_meters}m) may result in limited or no data. "
            "Consider using a radius of at least 50m for meaningful analysis."
        )
        if verbose:
            print(f"Warning: Very small radius ({radius_meters}m) may result in limited data.")
    elif radius_meters < 100:
        logger.info(f"Small radius ({radius_meters}m) may result in limited data.")
        if verbose:
            print(f"Info: Small radius ({radius_meters}m) may result in limited data.")
    
    try:
        # Create extractor with clean interface
        extractor = UrbanFeatureExtractor(
            radius_meters=radius_meters,
            use_cache=cache,
            verbose=verbose,
            progress_callback=progress_callback
        )
        
        # Extract features
        return extractor.extract_features(latitude, longitude, radius_meters)
        
    except Exception as e:
        logger.error(f"Failed to extract features for ({latitude}, {longitude}): {e}")
        raise GeoFeatureKitError(f"Error extracting features: {str(e)}") from e


def extract_features_batch(
    locations: List[Tuple[float, float, int]],
    *,
    verbose: bool = False,
    cache: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> List[Dict[str, Any]]:
    """Extract urban features for multiple locations efficiently.
    
    Args:
        locations: List of (latitude, longitude, radius_meters) tuples
        verbose: Enable verbose output to console (default: False)
        cache: Whether to use caching for downloaded data (default: True)
        progress_callback: Optional callback function(message: str, progress: float)
                          for overall batch progress
        
    Returns:
        List of feature dictionaries in the same order as input locations
        
    Raises:
        ValueError: If any location has invalid coordinates
        GeoFeatureKitError: If extraction fails for all locations
        
    Example:
        >>> locations = [
        ...     (40.7580, -73.9855, 500),  # Times Square
        ...     (40.7829, -73.9654, 500),  # Central Park
        ... ]
        >>> results = extract_features_batch(locations)
    """
    if not locations:
        return []
    
    results = []
    errors = []
    
    for i, (lat, lon, radius) in enumerate(locations):
        if progress_callback:
            progress = (i / len(locations))
            progress_callback(f"Processing location {i+1}/{len(locations)}", progress)
        
        try:
            result = extract_features(
                lat, lon, radius,
                verbose=False,  # Suppress individual verbose output in batch
                cache=cache,
                progress_callback=None  # Suppress individual progress in batch
            )
            results.append(result)
            
        except Exception as e:
            error_msg = f"Location {i+1} ({lat}, {lon}): {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            results.append(None)  # Placeholder for failed extraction
    
    if progress_callback:
        progress_callback("Batch processing complete", 1.0)
    
    if errors:
        if len(errors) == len(locations):
            raise GeoFeatureKitError("All locations failed to process")
        elif verbose:
            print(f"Warning: {len(errors)} of {len(locations)} locations failed to process")
    
    return results


def extract_multimodal_features(
    latitude: float,
    longitude: float,
    *,
    radius_meters: Optional[int] = None,
    walk_time_minutes: Optional[float] = None,
    bike_time_minutes: Optional[float] = None,
    drive_time_minutes: Optional[float] = None,
    speed_config: Optional[Dict[str, float]] = None,
    verbose: bool = False,
    cache: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Any]:
    """Extract urban features using multiple analysis methods.
    
    Supports both radius-based and isochrone-based analysis for walking,
    biking, and driving accessibility.
    
    Args:
        latitude: Location latitude (-90 to 90)
        longitude: Location longitude (-180 to 180)
        radius_meters: Optional radius for circular analysis
        walk_time_minutes: Optional maximum walking time
        bike_time_minutes: Optional maximum biking time  
        drive_time_minutes: Optional maximum driving time
        speed_config: Optional speed config with 'walk', 'bike', 'drive' keys (km/h)
        verbose: Enable verbose output (default: False)
        cache: Whether to use caching (default: True)
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary with keys for each analysis type requested:
        - radius_features: Features within circular radius (if radius_meters specified)
        - walk_features: Features within walking isochrone (if walk_time_minutes specified)
        - bike_features: Features within biking isochrone (if bike_time_minutes specified)
        - drive_features: Features within driving isochrone (if drive_time_minutes specified)
        
    Raises:
        ValueError: If coordinates invalid or no analysis type specified
        
    Example:
        >>> features = extract_multimodal_features(
        ...     40.7580, -73.9855,
        ...     radius_meters=500,
        ...     walk_time_minutes=10,
        ...     bike_time_minutes=5
        ... )
        >>> print(features.keys())
        dict_keys(['radius_features', 'walk_features', 'bike_features'])
    """
    # Validate coordinates
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90.")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180.")
    
    # Check that at least one analysis type is specified
    analysis_types = [radius_meters, walk_time_minutes, bike_time_minutes, drive_time_minutes]
    if all(x is None for x in analysis_types):
        raise ValueError("At least one analysis type must be specified")
    
    # Initialize speed configuration
    if speed_config is None:
        speed_config = get_default_speed_config()
    else:
        validate_speed_config(speed_config)
    
    results = {}
    total_analyses = sum(1 for x in analysis_types if x is not None)
    current_analysis = 0
    
    def make_progress_callback(analysis_name: str) -> Optional[ProgressCallback]:
        """Create a progress callback for individual analysis."""
        if progress_callback is None:
            return None
            
        def wrapped_callback(message: str, progress: float) -> None:
            overall_progress = (current_analysis + progress) / total_analyses
            progress_callback(f"{analysis_name}: {message}", overall_progress)
        
        return wrapped_callback
    
    try:
        # Radius-based features
        if radius_meters is not None:
            if radius_meters <= 0:
                raise ValueError("Radius must be positive")
            
            results['radius_features'] = extract_features(
                latitude, longitude, int(radius_meters),
                verbose=verbose,
                cache=cache,
                progress_callback=make_progress_callback("Radius analysis")
            )
            current_analysis += 1
        
        # Walking isochrone features
        if walk_time_minutes is not None:
            if walk_time_minutes <= 0:
                raise ValueError("Walk time must be positive")
            
            results['walk_features'] = extract_isochrone_features(
                latitude=latitude,
                longitude=longitude,
                travel_time_minutes=walk_time_minutes,
                mode='walk',
                speed_kmh=speed_config['walk'],
                verbose=verbose,
                progress_callback=make_progress_callback("Walking isochrone")
            )
            current_analysis += 1
        
        # Biking isochrone features  
        if bike_time_minutes is not None:
            if bike_time_minutes <= 0:
                raise ValueError("Bike time must be positive")
            
            results['bike_features'] = extract_isochrone_features(
                latitude=latitude,
                longitude=longitude,
                travel_time_minutes=bike_time_minutes,
                mode='bike',
                speed_kmh=speed_config['bike'],
                verbose=verbose,
                progress_callback=make_progress_callback("Biking isochrone")
            )
            current_analysis += 1
        
        # Driving isochrone features
        if drive_time_minutes is not None:
            if drive_time_minutes <= 0:
                raise ValueError("Drive time must be positive")
            
            results['drive_features'] = extract_isochrone_features(
                latitude=latitude,
                longitude=longitude,
                travel_time_minutes=drive_time_minutes,
                mode='drive',
                speed_kmh=speed_config['drive'],
                verbose=verbose,
                progress_callback=make_progress_callback("Driving isochrone")
            )
            current_analysis += 1
        
        if progress_callback:
            progress_callback("Multimodal analysis complete", 1.0)
        
        return results
        
    except Exception as e:
        logger.error(f"Failed multimodal extraction for ({latitude}, {longitude}): {e}")
        raise GeoFeatureKitError(f"Error extracting multimodal features: {str(e)}") from e 