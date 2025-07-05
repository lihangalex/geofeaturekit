"""Core feature extraction module."""

import warnings
import os
import networkx as nx
import geopandas as gpd
from typing import Dict, Any, List, Tuple, Optional
import osmnx as ox
from ..utils.network import download_network
from ..utils.poi import download_pois
from ..utils.network_advanced import download_land_use
from ..utils.area import calculate_area_hectares
from ..utils.formatting import round_float, AREA_DECIMALS
from ..exceptions.errors import GeoFeatureKitError
from .metrics import calculate_all_metrics

class UrbanFeatureExtractor:
    """Extract urban features from locations."""
    
    def __init__(self, radius_meters: int = 500, use_cache: bool = True):
        """Initialize extractor.
        
        Args:
            radius_meters: Analysis radius in meters
            use_cache: Whether to cache downloaded data
            
        Raises:
            GeoFeatureKitError: If radius is invalid
        """
        if radius_meters <= 0:
            raise GeoFeatureKitError("Radius must be positive")
            
        self.radius_meters = radius_meters
        self.use_cache = use_cache
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        self._cache = {}
        
        if use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _extract_network_features(self, G: nx.Graph) -> Dict[str, Any]:
        """Extract network features from graph."""
        return {
            'basic_metrics': {
                'total_street_length_meters': sum(d['length'] for _, _, d in G.edges(data=True)),
                'total_intersections': len([n for n, d in G.nodes(data=True) if d['street_count'] > 1]),
                'total_dead_ends': len([n for n, d in G.nodes(data=True) if d['street_count'] == 1]),
                'total_nodes': G.number_of_nodes(),
                'total_street_segments': G.number_of_edges()
            },
            'density_metrics': {},
            'connectivity_metrics': {},
            'street_pattern_metrics': {}
        }
    
    def _extract_poi_features(self, pois: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Extract POI features from GeoDataFrame."""
        return {
            'absolute_counts': {
                'total_points_of_interest': len(pois),
            },
            'density_metrics': {},
            'distribution_metrics': {}
        }
    
    def features_from_location(
        self,
        latitude: float,
        longitude: float,
        radius_meters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract urban features for a single location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
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
            return cached_data
        
        try:
            # Download data
            G = download_network(latitude, longitude, radius)
            pois = download_pois(latitude, longitude, radius)
            land_use = download_land_use(latitude, longitude, radius)
            
            # Calculate area in square meters
            area_sqm = calculate_area_hectares(radius) * 10000
            
            # Use the proper metrics calculation function
            metrics = calculate_all_metrics(G, pois, area_sqm)
            
            # Save to cache
            self._save_to_cache(cache_key, metrics)
            
            return metrics
            
        except Exception as e:
            raise GeoFeatureKitError(f"Failed to extract features: {str(e)}")
    
    def features_from_location_batch(
        self,
        locations: List[Tuple[float, float]],
        radius_meters: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Extract urban features for multiple locations.
        
        Args:
            locations: List of (latitude, longitude) tuples
            radius_meters: Analysis radius in meters. If None, uses default radius
            
        Returns:
            List of dictionaries containing urban features and metrics
            
        Raises:
            GeoFeatureKitError: If data download fails or metrics calculation fails
        """
        results = []
        errors = []
        
        for i, (lat, lon) in enumerate(locations):
            try:
                result = self.features_from_location(lat, lon, radius_meters)
                results.append(result)
            except Exception as e:
                errors.append(f"Location {i} ({lat}, {lon}): {str(e)}")
        
        if errors:
            warnings.warn(f"Errors occurred during batch processing:\n" + "\n".join(errors))
        
        return results
    
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