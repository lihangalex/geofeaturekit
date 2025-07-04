"""Configuration management for the GeoFeatureKit package."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import osmnx as ox
from pathlib import Path

# Configure OSMnx
ox.settings.log_console = False
ox.settings.use_cache = True

# Cache settings
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Default values
DEFAULT_RADIUS = 100  # meters
DEFAULT_CRS = "EPSG:4326"  # WGS 84
DEFAULT_OUTPUT_DIR = "output"  # Default directory for saving results

# POI categories and their associated OSM tags
POI_CATEGORIES = {
    "Transportation": {
        "amenity": ["bus_station", "parking"],
        "railway": ["station", "subway_entrance"],
        "public_transport": ["station", "stop_position"]
    },
    "Food & Drink": {
        "amenity": ["restaurant", "cafe", "bar", "pub", "fast_food"]
    },
    "Shopping": {
        "shop": ["supermarket", "mall", "convenience", "clothes", "electronics"],
        "amenity": ["marketplace"]
    },
    "Healthcare": {
        "amenity": ["hospital", "clinic", "pharmacy", "doctors"]
    },
    "Education": {
        "amenity": ["school", "university", "library", "college"]
    },
    "Recreation": {
        "leisure": ["park", "sports_centre", "fitness_centre", "playground"],
        "amenity": ["gym"]
    },
    "Cultural": {
        "tourism": ["museum", "gallery"],
        "amenity": ["theatre", "cinema", "arts_centre"]
    },
    "Services": {
        "amenity": ["bank", "post_office", "atm", "police"]
    },
    "Landmarks": {
        "tourism": ["attraction", "viewpoint", "monument"],
        "historic": ["monument", "memorial"]
    },
    "Other": {
        "amenity": ["place_of_worship", "community_centre"],
        "building": ["office"]
    }
}

@dataclass
class Config:
    """Base configuration class."""
    user_agent: str = "geofeaturekit"
    cache_enabled: bool = True
    log_to_console: bool = True
    save_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration settings."""
        if self.min_radius <= 0:
            raise ValueError("min_radius must be positive")
        if self.zoom_level < 0:
            raise ValueError("zoom_level must be non-negative")
        if not self.user_agent:
            raise ValueError("user_agent cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.sleep_time < 0:
            raise ValueError("sleep_time must be non-negative")
        if self.save_dir:
            save_path = Path(self.save_dir)
            if not save_path.parent.exists():
                raise ValueError(f"Parent directory for save_dir does not exist: {save_path.parent}")

# Default paths
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), 'cache')

# Default analysis settings
DEFAULT_RADIUS_METERS = 100  # Default radius for all analyses
DEFAULT_CRS = 'EPSG:4326'  # WGS84

# Feature computation settings
DEFAULT_EMBEDDING_DIMS = 128
DEFAULT_EMBEDDING_WALKS = 200
DEFAULT_EMBEDDING_WALK_LENGTH = 30

class AnalysisConfig:
    """Configuration for geospatial analysis."""
    
    def __init__(
        self,
        radius_meters: int = DEFAULT_RADIUS_METERS,
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        crs: str = DEFAULT_CRS
    ):
        """Initialize analysis configuration.
        
        Args:
            radius_meters: Global radius for all analyses in meters
            output_dir: Directory for saving output files
            cache_dir: Directory for caching network data
            crs: Coordinate reference system
            
        Raises:
            ValueError: If radius_meters is not positive
        """
        if radius_meters <= 0:
            raise ValueError("radius_meters must be positive")
            
        self.radius_meters = radius_meters
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.crs = crs
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True) 