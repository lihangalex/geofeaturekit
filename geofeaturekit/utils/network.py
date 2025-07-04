"""Network analysis utilities."""

import warnings
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Set, Optional, List, Any, Tuple
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from geopy.distance import geodesic

from geofeaturekit.core.config import DEFAULT_CRS
from geofeaturekit.utils.network_advanced import (
    compute_space_syntax_metrics,
    compute_orientation_entropy,
    compute_morphological_metrics,
    compute_hierarchical_metrics,
    compute_spectral_features,
    compute_advanced_stats
)
from ..utils.progress import create_progress_bar, log_analysis_start, log_analysis_complete

# Suppress the specific warning about great_circle_vec
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message='.*great_circle_vec.*')

def calculate_basic_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate basic network metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary containing basic network metrics
    """
    # Calculate total street length
    total_length = sum(d['length'] for _, _, d in G.edges(data=True))
    
    # Calculate intersection density
    intersections = len([n for n, d in G.degree() if d > 2])
    bounds = ox.utils_geo.bbox_from_point(
        (G.graph['center_lat'], G.graph['center_lon']),
        dist=G.graph['dist']
    )
    area_sqkm = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 111) ** 2
    intersection_density = intersections / area_sqkm if area_sqkm > 0 else 0
    
    # Calculate road type distribution
    road_types = {}
    for _, _, data in G.edges(data=True):
        road_type = process_road_type(data.get('highway'))
        length = float(data.get('length', 0))
        road_types[road_type] = road_types.get(road_type, 0) + length
    
    # Convert to percentages
    if total_length > 0:
        road_distribution = {k: float(v/total_length) for k, v in road_types.items()}
    else:
        road_distribution = {}
    
    # Calculate connectivity
    G_undirected = G.to_undirected()
    avg_degree = float(np.mean([d for _, d in G_undirected.degree()]))
    components = list(nx.connected_components(G_undirected))
    connectivity = {
        "avg_degree": avg_degree,
        "num_components": len(components),
        "largest_component_size": len(max(components, key=len))
    }
    
    return {
        "street_lengths": {
            "total_length": total_length,
            "avg_segment_length": total_length / G.number_of_edges() if G.number_of_edges() > 0 else 0
        },
        "intersection_density": {
            "intersections": intersections,
            "density": intersection_density
        },
        "road_types": road_distribution,
        "connectivity": connectivity
    }

def process_road_type(road_type) -> str:
    """Process road type value, handling both scalar and array inputs.
    
    Args:
        road_type: String or array-like of road type values
        
    Returns:
        str: The primary road type
    """
    if road_type is None:
        return "unknown"
        
    try:
        # Handle list-like objects without converting to numpy array
        if hasattr(road_type, '__iter__') and not isinstance(road_type, (str, bytes)):
            # Iterate directly over the values
            for val in road_type:
                if pd.notna(val):  # Use pandas NA check which handles all types
                    return str(val)
            return "unknown"
        else:
            # Handle scalar value
            return str(road_type) if pd.notna(road_type) else "unknown"
    except Exception as e:
        print(f"  Warning: Error processing road type: {str(e)}")
        return "unknown"

def get_network_stats(
    latitude: float, 
    longitude: float, 
    radius_meters: int
) -> Dict[str, Any]:
    """Get street network statistics using absolute metrics.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Analysis radius in meters
        
    Returns:
        Dictionary containing:
        - area_metrics: Size and basic area information
        - street_metrics: Street lengths and counts
        - network_metrics: Graph topology measures
        - geometric_metrics: Physical layout measurements
    """
    # Create point and get network
    point = (latitude, longitude)
    G = ox.graph_from_point(point, dist=radius_meters, network_type='all')
    
    # Convert to undirected for analysis
    G_undirected = G.to_undirected()
    
    # Calculate area in square kilometers
    area_sqkm = np.pi * (radius_meters / 1000) ** 2
    
    # Basic area metrics
    area_metrics = {
        "area_size_sq_km": area_sqkm,
        "center_point": {
            "latitude": float(latitude),
            "longitude": float(longitude)
        },
        "radius_meters": radius_meters
    }
    
    # Street metrics
    total_length = sum(float(d.get('length', 0)) for _, _, d in G.edges(data=True))
    intersections = len([n for n, d in G_undirected.degree() if d > 2])
    dead_ends = len([n for n, d in G_undirected.degree() if d == 1])
    
    street_metrics = {
        "total_street_length_meters": total_length,
        "intersections_count": intersections,
        "intersections_per_sq_km": intersections / area_sqkm if area_sqkm > 0 else 0,
        "dead_ends_count": dead_ends,
        "street_segments_count": G.number_of_edges(),
        "street_length_by_type_meters": _calculate_street_lengths_by_type(G)
    }
    
    # Network metrics
    network_metrics = {
        "average_node_degree": float(np.mean([d for _, d in G_undirected.degree()])),
        "total_node_count": G.number_of_nodes(),
        "total_edge_count": G.number_of_edges(),
        "connected_components_count": nx.number_connected_components(G_undirected),
        "largest_component_node_count": len(max(nx.connected_components(G_undirected), key=len))
    }
    
    # Geometric metrics
    geometric_metrics = {
        "street_angles_distribution_degrees": _calculate_angle_distribution(G),
        "average_segment_length_meters": total_length / G.number_of_edges() if G.number_of_edges() > 0 else 0,
        "block_sizes_meters": _calculate_block_sizes(G)
    }
    
    return {
        "area_metrics": area_metrics,
        "street_metrics": street_metrics,
        "network_metrics": network_metrics,
        "geometric_metrics": geometric_metrics
    }

def _calculate_street_lengths_by_type(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Calculate total street length by type in meters."""
    lengths = {}
    for _, _, data in G.edges(data=True):
        road_type = str(data.get('highway', 'unknown'))  # Convert to string to ensure hashable
        length = float(data.get('length', 0))
        lengths[road_type] = lengths.get(road_type, 0) + length
    return lengths

def _calculate_angle_distribution(G: nx.MultiDiGraph) -> Dict[str, int]:
    """Calculate distribution of street angles in 30-degree bins."""
    angles = []
    for _, _, data in G.edges(data=True):
        if 'bearing' in data:
            angle = float(data['bearing']) % 180
            angles.append(angle)
    
    bins = {
        "0-30": 0,
        "31-60": 0,
        "61-90": 0,
        "91-120": 0,
        "121-150": 0,
        "151-180": 0
    }
    
    for angle in angles:
        if angle <= 30:
            bins["0-30"] += 1
        elif angle <= 60:
            bins["31-60"] += 1
        elif angle <= 90:
            bins["61-90"] += 1
        elif angle <= 120:
            bins["91-120"] += 1
        elif angle <= 150:
            bins["121-150"] += 1
        else:
            bins["151-180"] += 1
    
    return bins

def _calculate_block_sizes(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Calculate block size statistics in meters."""
    block_lengths = [float(d.get('length', 0)) for _, _, d in G.edges(data=True) if 'length' in d]
    
    if not block_lengths:
        return {
            "min_length": 0,
            "max_length": 0,
            "mean_length": 0,
            "median_length": 0
        }
    
    return {
        "min_length": float(min(block_lengths)),
        "max_length": float(max(block_lengths)),
        "mean_length": float(np.mean(block_lengths)),
        "median_length": float(np.median(block_lengths))
    }

def calculate_network_length(G: nx.MultiDiGraph) -> float:
    """Calculate total length of the street network in meters.
    
    Args:
        G: NetworkX MultiDiGraph from OSMnx
        
    Returns:
        Total length in meters
    """
    return sum(float(d['length']) for _, _, d in G.edges(data=True))

def identify_intersections(G: nx.MultiDiGraph) -> List[Tuple[float, float]]:
    """Identify intersection nodes in the street network.
    
    Args:
        G: NetworkX MultiDiGraph from OSMnx
        
    Returns:
        List of (lat, lon) tuples for intersection nodes
    """
    intersections = []
    for node, degree in G.degree():
        if degree > 2:  # Nodes with more than 2 edges are intersections
            try:
                lat = G.nodes[node]['y']
                lon = G.nodes[node]['x']
                intersections.append((lat, lon))
            except KeyError:
                continue
    return intersections

def calculate_intersection_density(G: nx.MultiDiGraph) -> float:
    """Calculate intersection density (intersections per square kilometer).
    
    Args:
        G: NetworkX MultiDiGraph from OSMnx
        
    Returns:
        Intersection density (intersections/km²)
    """
    # Get network bounds
    bounds = ox.utils_geo.bbox_from_point(
        (G.graph['center_lat'], G.graph['center_lon']),
        dist=G.graph['dist']
    )
    area_sqkm = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 111) ** 2
    
    # Count intersections
    intersections = len(identify_intersections(G))
    
    return float(intersections / area_sqkm if area_sqkm > 0 else 0)

def calculate_network_density(G: nx.MultiDiGraph) -> float:
    """Calculate network density (total street length per square kilometer).
    
    Args:
        G: NetworkX MultiDiGraph from OSMnx
        
    Returns:
        Network density (meters/km²)
    """
    # Get network bounds
    bounds = ox.utils_geo.bbox_from_point(
        (G.graph['center_lat'], G.graph['center_lon']),
        dist=G.graph['dist']
    )
    area_sqkm = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 111) ** 2
    
    # Calculate total network length
    total_length = calculate_network_length(G)
    
    return float(total_length / area_sqkm if area_sqkm > 0 else 0)

def calculate_network_features(G, feature_sets=None):
    """
    Calculate network features for the given graph.
    
    Args:
        G: NetworkX graph object
        feature_sets: List of feature sets to compute. 
        Options: ['basic', 'pattern', 'spectral']
    
    Returns:
        Dictionary of computed features
    """
    if feature_sets is None:
        feature_sets = ['basic']
    
    results = {}
    
    # Basic features
    if 'basic' in feature_sets:
        results['basic'] = {
            'total_length': calculate_network_length(G),
            'intersection_density': calculate_intersection_density(G),
            'road_type_distribution': analyze_road_types(G),
            'connectivity': calculate_connectivity(G)
        }
    
    # Pattern features (formerly urban)
    if 'pattern' in feature_sets:
        results['pattern'] = {
            'grid_pattern_score': calculate_grid_pattern(G),
            'block_sizes': analyze_block_sizes(G),
            'street_orientation': analyze_street_orientation(G)
        }
    
    # Spectral features
    if 'spectral' in feature_sets:
        results['spectral'] = calculate_spectral_features(G)
    
    return results 