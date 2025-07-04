"""Core feature extraction functionality."""

from typing import Union, List, Dict, Optional, Set, Literal
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from .models import Location, AnalysisResults
from ..utils.network import get_network_stats, compute_contextual_embeddings_batch
from ..utils.poi import get_pois, process_pois
from ..core.config import AnalysisConfig, DEFAULT_RADIUS_METERS
from ..utils.progress import create_progress_bar, log_analysis_start, log_analysis_complete, log_error

class GeospatialFeatureExtractor:
    """Extract geospatial features for locations.
    
    Basic Usage:
        extractor = GeospatialFeatureExtractor()
        results = extractor.extract_features({
            'latitude': 40.7128,
            'longitude': -74.0060
        })
    
    Advanced Usage:
        extractor = GeospatialFeatureExtractor(
            enable_embeddings=True,
            embedding_dims=64,
            output_format='pandas',
            radius_meters=500
        )
    """
    
    def __init__(
        self,
        # Analysis settings
        radius_meters: int = DEFAULT_RADIUS_METERS,
        
        # Feature flags
        enable_embeddings: bool = False,
        enable_urban_metrics: bool = False,
        enable_spectral: bool = False,
        
        # Embedding parameters (only used if enable_embeddings=True)
        embedding_dims: int = 128,
        embedding_reduce_dims: Optional[int] = None,
        embedding_walks: int = 200,
        embedding_walk_length: int = 30,
        
        # Output parameters
        output_format: Literal['json', 'pandas', 'object'] = 'json',
        scale_features: bool = False,
        include_metadata: bool = True,
        
        # Optional configuration
        config: Optional[AnalysisConfig] = None
    ):
        """Initialize the extractor.
        
        Args:
            radius_meters: Global radius for all analyses in meters
            
            enable_embeddings: Whether to compute network embeddings (for ML)
            enable_urban_metrics: Whether to compute urban form metrics
            enable_spectral: Whether to compute spectral graph features
            
            embedding_dims: Number of dimensions for network embeddings
            embedding_reduce_dims: If set, reduce embeddings to this many dimensions
            embedding_walks: Number of random walks for Node2Vec
            embedding_walk_length: Length of random walks for Node2Vec
            
            output_format: Format of the output data
            scale_features: Whether to normalize numerical features
            include_metadata: Whether to include feature descriptions
            
            config: Optional configuration object (overrides other parameters)
        """
        # Use config if provided, otherwise create default
        self.config = config or AnalysisConfig(radius_meters=radius_meters)
        
        self._init_geocoder()
        
        # Store feature flags
        self.enable_embeddings = enable_embeddings
        self.enable_urban_metrics = enable_urban_metrics
        self.enable_spectral = enable_spectral
        
        # Store embedding config
        self.embedding_config = {
            'dimensions': embedding_dims,
            'reduce_dims': embedding_reduce_dims,
            'num_walks': embedding_walks,
            'walk_length': embedding_walk_length
        }
        
        # Store output config
        self.output_format = output_format
        self.scale_features = scale_features
        self.include_metadata = include_metadata
        
        # Set up feature sets based on flags
        self.feature_sets = {'basic'}  # Always include basic
        if enable_urban_metrics:
            self.feature_sets.add('urban')
        if enable_spectral:
            self.feature_sets.add('spectral')
    
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
        log_analysis_start(len(locations))
        
        # Pre-compute embeddings for all locations if enabled
        shared_embeddings = None
        if self.enable_embeddings:
            print("Computing embeddings...")
            shared_embeddings = compute_contextual_embeddings_batch(
                locations,
                self.config.radius_meters,
                dimensions=self.embedding_config['dimensions'],
                reduce_dims=self.embedding_config['reduce_dims'],
                num_walks=self.embedding_config['num_walks'],
                walk_length=self.embedding_config['walk_length']
            )
        
        results = []
        success_count = 0
        for loc in create_progress_bar(locations, desc="Processing locations"):
            try:
                result = self._extract_single(
                    loc, 
                    show_progress=False,
                    shared_embeddings=shared_embeddings
                )
                results.append(result)
                success_count += 1
            except Exception as e:
                log_error(f"{loc['latitude']}, {loc['longitude']}", e)
        
        log_analysis_complete(len(locations), success_count)
        return self._format_output(results, single=False)
    
    def _extract_single(
        self, 
        location: Dict[str, float], 
        show_progress: bool = False,
        shared_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> AnalysisResults:
        """Extract features for a single location."""
        lat, lon = location["latitude"], location["longitude"]
        
        if show_progress:
            print("  • Getting address information...", end="", flush=True)
        # Get address using reverse geocoding
        try:
            location_info = self.geolocator.reverse((lat, lon))
            address = location_info.address if location_info else "Address not found"
            if show_progress:
                print(" Done")
        except GeocoderTimedOut:
            address = "Address lookup timed out"
            if show_progress:
                print(" Failed (timeout)")
        
        if show_progress:
            print("  • Analyzing street network...", end="", flush=True)
        # Get network stats with specified features
        network_data = get_network_stats(
            lat, 
            lon, 
            self.config.radius_meters,
            feature_sets=self.feature_sets,
            compute_embeddings=self.enable_embeddings,
            embedding_config=self.embedding_config,
            shared_embeddings=shared_embeddings,
            location_key=f"{lat},{lon}"  # Use string format for location key
        )
        
        if show_progress:
            print(" Done")
        
        # Get POIs
        poi_data = get_pois(lat, lon, self.config.radius_meters)
        processed_pois = process_pois(poi_data)
        
        # Create location object
        loc = Location(latitude=lat, longitude=lon, address=address)
        
        return AnalysisResults(
            location=loc,
            radius=self.config.radius_meters,
            network_stats=network_data,  # Now using the raw dictionary
            points_of_interest=processed_pois
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