"""Core feature extraction module."""

import os
import logging
import networkx as nx
import geopandas as gpd
from typing import Dict, Any, List, Tuple, Optional, Callable
import osmnx as ox
from ..utils.network import download_network
from ..utils.poi import download_pois
from ..utils.network_advanced import download_land_use
from ..utils.area import calculate_area_hectares
from ..utils.formatting import round_float, AREA_DECIMALS
from ..utils.progress import EnhancedProgressTracker, BatchProgressTracker
from ..exceptions.errors import GeoFeatureKitError
from .metrics import calculate_all_metrics

# Set up logger
logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[str, float], None]


class UrbanFeatureExtractor:
    """Extract urban features from locations.
    
    This class provides the core functionality for extracting geospatial
    features from urban areas, including street network analysis and
    point of interest (POI) metrics.
    """
    
    def __init__(
        self, 
        radius_meters: int = 500, 
        use_cache: bool = True, 
        verbose: bool = False,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """Initialize the urban feature extractor.
        
        Args:
            radius_meters: Analysis radius in meters (default: 500)
            use_cache: Whether to cache downloaded data (default: True)
            verbose: Enable verbose console output (default: False) 
            progress_callback: Optional callback for progress updates
                             (message: str, progress: float) where progress is 0.0-1.0
            
        Raises:
            GeoFeatureKitError: If radius is invalid
        """
        if radius_meters <= 0:
            raise GeoFeatureKitError("Radius must be positive")
        
        # Log warnings for small radii instead of printing
        if radius_meters < 10:
            logger.warning(
                f"Extremely small radius ({radius_meters}m) will likely result in no street network data. "
                "Network metrics will be null, but POI and land use data may still be available."
            )
            if verbose:
                print(f"Warning: Extremely small radius ({radius_meters}m) may result in no data.")
        elif radius_meters < 50:
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
            
        self.radius_meters = radius_meters
        self.use_cache = use_cache
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        self._cache = {}
        
        if use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def extract_features(
        self,
        latitude: float,
        longitude: float,
        radius_meters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract urban features for a single location.
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180) 
            radius_meters: Analysis radius in meters. If None, uses default radius
            
        Returns:
            Dictionary containing urban features and metrics
            
        Raises:
            ValueError: If coordinates are invalid
            GeoFeatureKitError: If data download fails or metrics calculation fails
        """
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90.")
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180.")
        
        radius = radius_meters or self.radius_meters
        
        # Check cache first
        cache_key = self._get_cache_key(latitude, longitude, radius)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            if self.verbose:
                print(f"Using cached data for ({latitude:.4f}, {longitude:.4f})")
            return cached_data
        
        # Create a progress tracker that respects our new API
        progress_tracker = self._create_progress_tracker(
            f"({latitude:.4f}, {longitude:.4f})"
        )
        
        try:
            if self.progress_callback:
                self.progress_callback("Starting feature extraction", 0.0)
            
            # Phase 1: Data Download (70% of total time)
            if self.progress_callback:
                self.progress_callback("Downloading street network", 0.1)
            elif self.verbose:
                print("Downloading street network...")
            
            G = download_network(latitude, longitude, radius)
            
            if self.progress_callback:
                self.progress_callback("Downloading points of interest", 0.4)
            elif self.verbose:
                print("Downloading points of interest...")
                
            pois = download_pois(latitude, longitude, radius)
            
            if self.progress_callback:
                self.progress_callback("Downloading land use data", 0.6)
            elif self.verbose:
                print("Downloading land use data...")
                
            land_use = download_land_use(latitude, longitude, radius)
            
            # Phase 2: Calculate Metrics (25% of total time)
            if self.progress_callback:
                self.progress_callback("Calculating area", 0.7)
            elif self.verbose:
                print("Calculating metrics...")
                
            area_sqm = calculate_area_hectares(radius) * 10000
            
            if self.progress_callback:
                self.progress_callback("Computing urban metrics", 0.8)
                
            metrics = calculate_all_metrics(G, pois, area_sqm)
            
            # Phase 3: Format Results (5% of total time)
            if self.progress_callback:
                self.progress_callback("Finalizing results", 0.9)
                
            # Save to cache
            self._save_to_cache(cache_key, metrics)
            
            if self.progress_callback:
                self.progress_callback("Complete", 1.0)
            elif self.verbose:
                print("Feature extraction complete")
                
            return metrics
            
        except Exception as e:
            error_msg = f"Failed to extract features for ({latitude:.4f}, {longitude:.4f}): {str(e)}"
            logger.error(error_msg)
            if self.progress_callback:
                self.progress_callback(f"Error: {str(e)}", 1.0)
            raise GeoFeatureKitError(f"Failed to extract features: {str(e)}") from e
    
    def extract_features_batch(
        self,
        locations: List[Tuple[float, float, int]],
    ) -> List[Dict[str, Any]]:
        """Extract urban features for multiple locations efficiently.
        
        Args:
            locations: List of (latitude, longitude, radius_meters) tuples
            
        Returns:
            List of dictionaries containing urban features and metrics
            
        Raises:
            GeoFeatureKitError: If data download fails or metrics calculation fails
        """
        results = []
        errors = []
        
        if self.verbose:
            print(f"Processing {len(locations)} locations...")
        
        for i, (lat, lon, radius) in enumerate(locations):
            if self.progress_callback:
                progress = i / len(locations)
                self.progress_callback(f"Processing location {i+1}/{len(locations)}", progress)
            elif self.verbose:
                print(f"Processing location {i+1}/{len(locations)}: ({lat:.4f}, {lon:.4f})")
            
            try:
                # Temporarily disable individual progress for batch processing
                original_callback = self.progress_callback
                original_verbose = self.verbose
                self.progress_callback = None
                self.verbose = False
                
                result = self.extract_features(lat, lon, radius)
                results.append(result)
                
            except Exception as e:
                error_msg = f"Location {i+1} ({lat}, {lon}): {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                results.append(None)  # Placeholder for failed extraction
                
            finally:
                # Restore original settings
                self.progress_callback = original_callback
                self.verbose = original_verbose
        
        if self.progress_callback:
            self.progress_callback("Batch processing complete", 1.0)
        elif self.verbose:
            print("Batch processing complete")
        
        if errors:
            if len(errors) == len(locations):
                raise GeoFeatureKitError("All locations failed to process")
            elif self.verbose:
                print(f"Warning: {len(errors)} of {len(locations)} locations failed to process")
        
        return results
    
    def _create_progress_tracker(self, location_desc: str):
        """Create appropriate progress tracker based on settings."""
        if self.progress_callback or self.verbose:
            return EnhancedProgressTracker(
                show_progress=self.verbose,
                detail_level='normal'
            )
        return None
    
    def _get_cache_key(self, latitude: float, longitude: float, radius: int) -> str:
        """Generate cache key for a location."""
        return f"{latitude},{longitude},{radius}"
    
    def _load_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if available."""
        if not self.use_cache:
            return None
        return self._cache.get(key)
    
    def _save_to_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        if self.use_cache:
            self._cache[key] = data 