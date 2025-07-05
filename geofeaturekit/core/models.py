"""Data models for the GeoFeatureKit package."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any
import json
import pandas as pd
import numpy as np

@dataclass
class Location:
    """Location data."""
    latitude: float
    longitude: float
    address: str
    
    def __post_init__(self):
        """Validate location data."""
        if not isinstance(self.latitude, (int, float)):
            raise ValueError("Latitude must be a number")
        if not isinstance(self.longitude, (int, float)):
            raise ValueError("Longitude must be a number")
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Location':
        """Create a Location from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
        
        required = {'latitude', 'longitude', 'address'}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required fields: {missing}")
            
        return cls(
            latitude=data['latitude'],
            longitude=data['longitude'],
            address=data['address']
        )
    
    @classmethod
    def from_input(cls, data: Union[dict, List[dict]]) -> Union['Location', List['Location']]:
        """Create Location(s) from input data."""
        if isinstance(data, dict):
            return cls.from_dict(data)
        elif isinstance(data, list):
            return [cls.from_dict(item) for item in data]
        else:
            raise ValueError("Input must be a dictionary or list of dictionaries")

@dataclass
class NetworkStats:
    """Network statistics."""
    total_street_length_meters: float
    intersections: int
    street_segments: int
    centrality: Dict[str, float]
    road_distribution: Dict[str, float]
    density: float
    # Urban form metrics
    space_syntax: Optional[Dict[str, float]] = None
    orientation: Optional[Dict[str, float]] = None
    morphology: Optional[Dict[str, float]] = None
    hierarchy: Optional[Dict[str, Dict[str, float]]] = None
    # Connectivity patterns
    connectivity_patterns: Optional[List[float]] = None
    # ML embeddings
    contextual_embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_ml_features(self) -> Dict[str, Any]:
        """Get features for machine learning."""
        features = {
            'streets': {
                'total_length': self.total_street_length_meters,
                'intersections': self.intersections,
                'segments': self.street_segments,
                'density': self.density
            },
            'centrality': self.centrality,
            'road_distribution': self.road_distribution
        }
        
        if self.space_syntax:
            features['urban_form'] = {
                'space_syntax': self.space_syntax,
                'orientation': self.orientation,
                'morphology': self.morphology,
                'hierarchy': self.hierarchy
            }
        
        if self.connectivity_patterns:
            features['connectivity'] = {
                'patterns': self.connectivity_patterns
            }
            
        if self.contextual_embeddings:
            features['embeddings'] = self.contextual_embeddings
            
        return features

@dataclass
class AnalysisResults:
    """Results from geospatial feature analysis."""
    location: Location
    radius: float
    network_stats: Dict[str, Any]
    points_of_interest: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert results to dictionary format."""
        def handle_nan(obj):
            if isinstance(obj, dict):
                return {k: handle_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [handle_nan(item) for item in obj]
            elif pd.isna(obj) or obj is None:
                return None
            return obj
        
        result = {
            'location': {
                'latitude': self.location.latitude,
                'longitude': self.location.longitude,
                'address': self.location.address
            },
            'radius': self.radius,
            'network_stats': self.network_stats,
            'points_of_interest': self.points_of_interest
        }
        
        # Process POI names
        if 'details' in result['points_of_interest']:
            for category, items in result['points_of_interest']['details'].items():
                for item in items:
                    if item.get('name') is None or pd.isna(item.get('name')):
                        # Use the category name from the parent key
                        item['name'] = f"Unnamed {category.replace('_', ' ').title()}"
        
        return handle_nan(result)
    
    def to_features(self) -> dict:
        """Convert results to feature vector format."""
        features = {}
        
        # Add network features
        if self.network_stats:
            features.update({
                'network_length': self.network_stats.get('basic', {}).get('total_length', 0),
                'intersection_count': self.network_stats.get('basic', {}).get('intersections', 0),
                'segment_count': self.network_stats.get('basic', {}).get('segments', 0)
            })
            
            # Add centrality metrics if available
            centrality = self.network_stats.get('basic', {}).get('centrality', {})
            if centrality:
                features.update({
                    'avg_betweenness': centrality.get('avg_betweenness', 0),
                    'avg_closeness': centrality.get('avg_closeness', 0),
                    'avg_degree': centrality.get('avg_degree', 0)
                })
            
            # Add road distribution if available
            road_dist = self.network_stats.get('basic', {}).get('road_distribution', {})
            if road_dist:
                features.update({
                    f'road_pct_{road_type}': pct 
                    for road_type, pct in road_dist.items()
                })
        
        # Add POI metrics
        if self.points_of_interest:
            features['poi_metrics'] = self.points_of_interest.get('counts', {})
        
        return features
    
    def to_json(self, indent: int = 2) -> str:
        """Convert results to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = []
        
        # Location info
        lines.append(f"Location: {self.location.latitude:.4f}, {self.location.longitude:.4f}")
        if self.location.address:
            lines.append(f"Address: {self.location.address}")
        lines.append(f"Analysis radius: {self.radius}m")
        
        # Network stats
        if self.network_stats:
            lines.append("\nNetwork Statistics:")
            basic_stats = self.network_stats.get('basic', {})
            lines.append(f"- Total street length: {basic_stats.get('total_length', 0):.1f}m")
            lines.append(f"- Intersections: {basic_stats.get('intersections', 0)}")
            lines.append(f"- Street segments: {basic_stats.get('segments', 0)}")
        
        # POI summary
        if self.points_of_interest:
            lines.append("\nPoints of Interest:")
            poi_counts = self.points_of_interest.get('counts', {})
            if not poi_counts:
                lines.append("- No POIs found")
            else:
                # Sort POIs by count
                for category, count in sorted(poi_counts.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {category}: {count}")
                
                # Add detailed POI information
                poi_details = self.points_of_interest.get('details', {})
                if poi_details:
                    lines.append("\nDetailed POI Information:")
                    for category, items in poi_details.items():
                        lines.append(f"\n{category.title()}:")
                        for item in items:
                            name = item.get('name') or f"Unnamed {category.replace('_', ' ').title()}"
                            lat = item.get('latitude', 0)
                            lon = item.get('longitude', 0)
                            lines.append(f"- {name} ({lat:.4f}, {lon:.4f})")
        
        return "\n".join(lines)

def results_to_dataframe(results_list: List[AnalysisResults]) -> pd.DataFrame:
    """Convert a list of AnalysisResults to a pandas DataFrame."""
    return pd.DataFrame([result.to_dict() for result in results_list]) 