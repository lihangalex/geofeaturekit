"""Core feature extraction functionality."""

from typing import Union, List, Dict, Optional, Set, Literal, Any
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import osmnx as ox
import networkx as nx

from .models import Location, AnalysisResults
from ..utils.network import get_network_stats
from ..utils.poi import analyze_pois
from ..core.config import AnalysisConfig, DEFAULT_RADIUS_METERS
from ..utils.progress import ProgressTracker

class GeospatialFeatureExtractor:
    """Extract geospatial features for locations using absolute metrics."""
    
    def __init__(
        self,
        # Analysis settings
        radius_meters: int = DEFAULT_RADIUS_METERS,
        
        # Output parameters
        output_format: Literal['json', 'pandas', 'object'] = 'json',
        include_metadata: bool = True,
        
        # Optional configuration
        config: Optional[AnalysisConfig] = None
    ):
        """Initialize the extractor.
        
        Args:
            radius_meters: Analysis radius in meters
            output_format: Format of the output data
            include_metadata: Whether to include feature descriptions
            config: Optional configuration object
        """
        self.config = config or AnalysisConfig(radius_meters=radius_meters)
        self._init_geocoder()
        
        # Store output config
        self.output_format = output_format
        self.include_metadata = include_metadata
    
    def _init_geocoder(self):
        """Initialize the geocoding service."""
        self.geolocator = Nominatim(user_agent="geofeaturekit")
    
    def extract_features(
        self,
        locations: Union[Dict[str, float], List[Dict[str, float]]]
    ) -> Union[AnalysisResults, List[AnalysisResults], Dict, List[Dict], pd.DataFrame]:
        """Extract features for one or more locations."""
        # Handle single location
        if isinstance(locations, dict):
            print("\nAnalyzing single location...")
            result = self._extract_single(locations, show_progress=True)
            return self._format_output([result], single=True)
        
        # Handle multiple locations
        print("\nAnalyzing multiple locations...")
        results = []
        for loc in tqdm(locations, desc="Processing locations"):
            try:
                result = self._extract_single(loc, show_progress=False)
                results.append(result)
            except Exception as e:
                print(f"Error processing {loc['latitude']}, {loc['longitude']}: {str(e)}")
        
        return self._format_output(results, single=False)
    
    def _extract_single(
        self, 
        location: Dict[str, float],
        show_progress: bool = False
    ) -> AnalysisResults:
        """Extract features for a single location using absolute metrics."""
        lat, lon = location["latitude"], location["longitude"]
        
        # Get address using reverse geocoding
        try:
            location_info = self.geolocator.reverse((lat, lon))
            address = location_info.address if location_info else "Address not found"
        except GeocoderTimedOut:
            address = "Address lookup timed out"
        
        # Create location object
        loc = Location(latitude=lat, longitude=lon, address=address)
        
        # Get network stats with absolute metrics
        network_data = get_network_stats(
            lat, 
            lon, 
            self.config.radius_meters
        )
        
        # Get POI data with absolute counts
        poi_data = analyze_pois(lat, lon, self.config.radius_meters)
        
        return AnalysisResults(
            location=loc,
            radius=self.config.radius_meters,
            network_stats=network_data,
            points_of_interest=poi_data
        )
    
    def _format_output(
        self, 
        results: List[AnalysisResults],
        single: bool = False
    ) -> Union[AnalysisResults, List[AnalysisResults], Dict, List[Dict], pd.DataFrame]:
        """Format results according to output configuration."""
        if self.output_format == 'json':
            formatted = [result.to_dict() for result in results]
            return formatted[0] if single else formatted
        
        if self.output_format == 'pandas':
            # Convert to DataFrame for pandas or further processing
            features_list = []
            for result in results:
                features = result.to_features()
                
                # Add location information
                features.update({
                    'latitude': result.location.latitude,
                    'longitude': result.location.longitude,
                    'address': result.location.address,
                    'radius': result.radius
                })
                
                features_list.append(features)
            
            return pd.DataFrame(features_list)
        
        # Return raw objects
        return results[0] if single else results 

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor."""
        self.progress = ProgressTracker()
    
    def extract_features(
        self, 
        location: str,
        # Network features (enabled by default)
        include_street_lengths: bool = True,
        include_intersection_density: bool = True,
        include_road_types: bool = True,
        include_connectivity: bool = True,
        
        # Pattern features
        include_grid_pattern: bool = False,
        include_block_sizes: bool = False,
        include_street_orientation: bool = False,
        
        # Accessibility features
        include_centrality: bool = False,
        include_reach: bool = False,
        include_betweenness: bool = False,
        
        # POI features
        include_poi_density: bool = False,
        include_poi_diversity: bool = False,
        include_poi_categories: bool = False,
        
        # Output settings
        save_visualization: bool = False,
        output_folder: str = "output"
    ) -> Dict[str, Any]:
        """
        Extract features based on enabled flags.
        
        Args:
            location: Name or address of the area to analyze
            include_*: Boolean flags for each feature
            save_visualization: Whether to save network visualization
            output_folder: Where to save output files
        
        Returns:
            Dictionary of computed features
        """
        self.progress.start("Loading network data")
        G = ox.graph_from_place(location, network_type="drive")
        features = {}
        
        # Basic network metrics
        if any([
            include_street_lengths,
            include_intersection_density,
            include_road_types,
            include_connectivity
        ]):
            self.progress.update("Calculating network metrics")
            network_metrics = calculate_basic_metrics(G)
            
            if include_street_lengths:
                features["street_lengths"] = network_metrics["street_lengths"]
            if include_intersection_density:
                features["intersection_density"] = network_metrics["intersection_density"]
            if include_road_types:
                features["road_types"] = network_metrics["road_types"]
            if include_connectivity:
                features["connectivity"] = network_metrics["connectivity"]
        
        # Pattern metrics
        if any([
            include_grid_pattern,
            include_block_sizes,
            include_street_orientation
        ]):
            self.progress.update("Analyzing street patterns")
            pattern_metrics = calculate_advanced_metrics(G)
            
            if include_grid_pattern:
                features["grid_pattern"] = pattern_metrics["grid_pattern"]
            if include_block_sizes:
                features["block_sizes"] = pattern_metrics["block_sizes"]
            if include_street_orientation:
                features["street_orientation"] = pattern_metrics["street_orientation"]
        
        # Accessibility metrics
        if any([
            include_centrality,
            include_reach,
            include_betweenness
        ]):
            self.progress.update("Computing accessibility metrics")
            accessibility_metrics = nx.algorithms.centrality.betweenness_centrality(G)
            
            if include_centrality:
                features["centrality"] = accessibility_metrics
            if include_reach:
                features["reach"] = self._calculate_reach(G)
            if include_betweenness:
                features["betweenness"] = accessibility_metrics
        
        # POI analysis
        if any([
            include_poi_density,
            include_poi_diversity,
            include_poi_categories
        ]):
            self.progress.update("Analyzing points of interest")
            poi_metrics = analyze_pois(location)
            
            if include_poi_density:
                features["poi_density"] = poi_metrics["density"]
            if include_poi_diversity:
                features["poi_diversity"] = poi_metrics["diversity"]
            if include_poi_categories:
                features["poi_categories"] = poi_metrics["categories"]
        
        # Save results
        if features:  # Only save if we have features
            output_dir = Path(output_folder)
            output_dir.mkdir(exist_ok=True)
            
            # Create a clean filename from the location
            filename = f"{location.replace(' ', '_').lower()}_features.json"
            output_path = output_dir / filename
            
            with open(output_path, 'w') as f:
                json.dump(features, f, indent=2)
            
            if save_visualization:
                self._save_visualization(G, location, output_dir)
        
        self.progress.complete()
        return features
    
    def _calculate_reach(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate reach metrics for the network."""
        # Implementation details for reach calculation
        return {"reach_500m": 0.0, "reach_1000m": 0.0}  # Placeholder
    
    def _save_visualization(self, G: nx.Graph, location: str, output_dir: Path):
        """Save a visualization of the network."""
        fig, ax = ox.plot_graph(G, show=False, close=False)
        output_path = output_dir / f"{location.replace(' ', '_').lower()}_network.png"
        fig.savefig(output_path)
        fig.close() 