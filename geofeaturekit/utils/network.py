"""Network analysis utilities with focus on physical measurements."""

import warnings
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
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
from ..utils.progress import create_progress_bar, log_analysis_start, log_analysis_complete, log_error
from .area import calculate_area_hectares
from .formatting import (
    round_float,
    LENGTH_DECIMALS,
    AREA_DECIMALS,
    DENSITY_DECIMALS,
    RATIO_DECIMALS,
    ANGLE_DECIMALS
)
from ..exceptions.errors import GeoFeatureKitError

# Suppress the specific warning about great_circle_vec
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message='.*great_circle_vec.*')

def calculate_basic_metrics(G: nx.MultiDiGraph, area_hectares: float) -> Dict[str, Any]:
    """Calculate basic network metrics.
    
    Args:
        G: NetworkX graph
        area_hectares: Area in hectares
        
    Returns:
        Dictionary containing basic metrics
    """
    # Calculate basic stats
    total_length = sum(d["length"] for _, _, d in G.edges(data=True))
    total_length_km = total_length / 1000
    intersections = len([n for n, d in G.degree() if d > 2])
    
    # Calculate intersection density (per hectare)
    intersection_density = intersections / area_hectares if area_hectares > 0 else 0
    
    # Calculate street density (km per hectare)
    street_density = total_length_km / area_hectares if area_hectares > 0 else 0
    
    return {
        "total_street_length_meters": round_float(total_length, LENGTH_DECIMALS),
        "total_street_length_km": round_float(total_length_km, LENGTH_DECIMALS),
        "total_intersections": intersections,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "density": {
            "intersections_per_hectare": round_float(intersection_density, DENSITY_DECIMALS),
            "street_km_per_hectare": round_float(street_density, DENSITY_DECIMALS),
            "units": "per_hectare"
        }
    }

def process_road_type(road_type) -> str:
    """Process road type string to standardized format."""
    if not road_type:
        return "unknown"
    if isinstance(road_type, list):
        road_type = road_type[0]
    return str(road_type).lower()

def _classify_nodes(G: nx.MultiDiGraph) -> Tuple[int, int, int]:
    """Classify nodes into intersections, dead ends, and bends.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Tuple of (intersections, dead_ends, bends)
    """
    intersections = 0
    dead_ends = 0
    bends = 0
    
    for _, degree in G.degree():
        if degree > 2:
            intersections += 1
        elif degree == 1:
            dead_ends += 1
        else:
            bends += 1
            
    return intersections, dead_ends, bends

def get_network_stats(
    latitude: float, 
    longitude: float, 
    radius_meters: int
) -> Dict[str, Any]:
    """Get street network statistics using physical measurements.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Analysis radius in meters
        
    Returns:
        Dictionary containing:
        - area_metrics: Physical size and coverage
        - street_metrics: Street measurements and patterns
        - network_metrics: Connectivity and accessibility
        - pattern_metrics: Street layout analysis
    """
    # Create point and get network
    point = (latitude, longitude)
    G = ox.graph_from_point(point, dist=radius_meters, network_type='all')
    
    # Calculate area in hectares
    area_hectares = calculate_area_hectares(radius_meters)
    
    # Basic area metrics
    area_metrics = {
        "area_hectares": round_float(area_hectares, AREA_DECIMALS),
        "center_point": {
            "latitude": float(latitude),
            "longitude": float(longitude)
        },
        "radius_meters": radius_meters,
        "edge_length_meters": 2 * np.pi * radius_meters  # Perimeter of analysis area
    }
    
    # Get detailed street metrics
    street_metrics = _calculate_street_metrics(G, area_hectares)
    
    # Get network connectivity metrics
    network_metrics = _calculate_network_metrics(G)
    
    # Get street pattern metrics
    pattern_metrics = _calculate_pattern_metrics(G)
    
    return {
        "area_metrics": area_metrics,
        "street_metrics": street_metrics,
        "network_metrics": network_metrics,
        "pattern_metrics": pattern_metrics
    }

def _calculate_street_metrics(G: nx.MultiDiGraph, area_hectares: float) -> Dict[str, Any]:
    """Calculate street metrics.
    
    Args:
        G: NetworkX graph
        area_hectares: Area in hectares
        
    Returns:
        Dictionary of street metrics
    """
    # Get all street segments
    street_lengths = [float(d['length']) for _, _, d in G.edges(data=True)]
    
    # Calculate basic length statistics
    total_length = sum(street_lengths)
    total_length_km = total_length / 1000
    avg_segment_length = np.mean(street_lengths) if street_lengths else 0
    
    # Classify nodes
    intersections, dead_ends, bends = _classify_nodes(G)
    
    # Calculate densities
    street_density = total_length_km / area_hectares if area_hectares > 0 else 0
    intersection_density = intersections / area_hectares if area_hectares > 0 else 0
    
    return {
        "length_metrics": {
            "total_street_length_meters": float(total_length),
            "total_street_length_km": float(total_length_km),
            "average_segment_length_meters": float(avg_segment_length),
            "street_length_stats": {
                "min_meters": float(min(street_lengths)) if street_lengths else 0,
                "max_meters": float(max(street_lengths)) if street_lengths else 0,
                "median_meters": float(np.median(street_lengths)) if street_lengths else 0,
                "std_meters": float(np.std(street_lengths)) if street_lengths else 0
            }
        },
        "density": {
            "intersections_per_hectare": round_float(intersection_density, DENSITY_DECIMALS),
            "street_km_per_hectare": round_float(street_density, DENSITY_DECIMALS),
            "units": "per_hectare"
        },
        "node_metrics": {
            "intersections": intersections,
            "dead_ends": dead_ends,
            "bends": bends,
            "total_nodes": G.number_of_nodes()
        }
    }

def _calculate_network_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate network metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of network metrics
    """
    # Calculate basic connectivity metrics
    G_undirected = G.to_undirected()
    avg_degree = float(np.mean([d for _, d in G_undirected.degree()]))
    components = list(nx.connected_components(G_undirected))
    
    # Calculate network efficiency
    efficiency = 0.0
    if len(G) > 1:
        try:
            efficiency = nx.global_efficiency(G_undirected)
        except:
            pass
    
    return {
        "connectivity": {
            "average_degree": float(avg_degree),
            "number_of_components": len(components),
            "largest_component_size": len(max(components, key=len)) if components else 0,
            "network_efficiency": float(efficiency)
        }
    }

def _calculate_pattern_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate street pattern metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of pattern metrics
    """
    # Get all bearings
    bearings = []
    for _, _, data in G.edges(data=True):
        if 'bearing' in data:
            bearing = float(data['bearing'])
            bearings.append(bearing)
    
    if not bearings:
        return {
            "orientation": {
                "mean_bearing": 0.0,
                "bearing_entropy": 0.0
            },
            "grid_characteristics": {
                "grid_pattern_ratio": 0.0,
                "organic_pattern_ratio": 0.0
            }
        }
    
    # Calculate orientation metrics
    mean_bearing = float(np.mean(bearings))
    hist, _ = np.histogram(bearings, bins=36, range=(0, 360))
    hist = hist / len(bearings)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Calculate grid vs organic pattern
    grid_angles = [0, 90, 180, 270]
    grid_count = sum(1 for b in bearings if any(abs((b % 360) - a) < 10 for a in grid_angles))
    grid_ratio = grid_count / len(bearings)
    
    return {
        "orientation": {
            "mean_bearing": float(mean_bearing),
            "bearing_entropy": float(entropy)
        },
        "grid_characteristics": {
            "grid_pattern_ratio": float(grid_ratio),
            "organic_pattern_ratio": float(1 - grid_ratio)
        }
    }

def _calculate_street_types(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate street type distributions.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of street type metrics
    """
    type_lengths = {}
    total_length = 0
    
    for _, _, data in G.edges(data=True):
        length = float(data.get('length', 0))
        road_type = data.get('highway', 'unknown')
        if isinstance(road_type, list):
            road_type = road_type[0]
            
        type_lengths[road_type] = type_lengths.get(road_type, 0) + length
        total_length += length
    
    # Calculate percentages and organize by hierarchy
    hierarchy = {
        "primary": ["motorway", "trunk", "primary"],
        "secondary": ["secondary", "tertiary"],
        "local": ["residential", "living_street", "unclassified"],
        "other": ["service", "track", "path", "footway", "cycleway"]
    }
    
    result = {
        "by_type": {
            road_type: {
                "length_meters": float(length),
                "percentage": float(length / total_length * 100) if total_length > 0 else 0
            }
            for road_type, length in type_lengths.items()
        },
        "by_hierarchy": {
            level: {
                "length_meters": float(sum(type_lengths.get(rt, 0) for rt in types)),
                "percentage": float(sum(type_lengths.get(rt, 0) for rt in types) / total_length * 100) if total_length > 0 else 0
            }
            for level, types in hierarchy.items()
        }
    }
    
    return result

def _calculate_route_directness(G: nx.Graph) -> Dict[str, Any]:
    """Calculate route directness metrics.
    
    Args:
        G: NetworkX graph (should be connected)
        
    Returns:
        Dictionary of route metrics
    """
    if G.number_of_nodes() < 2:
        return {
            "average_directness": 0.0,
            "direct_routes_ratio": 0.0
        }
    
    # Sample node pairs if graph is large
    nodes = list(G.nodes())
    if len(nodes) > 100:
        np.random.seed(42)  # For reproducibility
        sample_size = min(100, len(nodes) * (len(nodes) - 1) // 2)
        pairs = [(nodes[i], nodes[j]) 
                for i in range(len(nodes)) 
                for j in range(i + 1, len(nodes))]
        pairs = np.random.choice(pairs, sample_size, replace=False)
    else:
        pairs = [(nodes[i], nodes[j]) 
                for i in range(len(nodes)) 
                for j in range(i + 1, len(nodes))]
    
    directness_ratios = []
    direct_routes = 0
    
    for start, end in pairs:
        try:
            # Get network distance
            path = nx.shortest_path_length(G, start, end, weight='length')
            
            # Get straight-line distance
            start_point = Point(G.nodes[start]['x'], G.nodes[start]['y'])
            end_point = Point(G.nodes[end]['x'], G.nodes[end]['y'])
            straight_dist = start_point.distance(end_point)
            
            if straight_dist > 0:
                ratio = path / straight_dist
                directness_ratios.append(ratio)
                if ratio < 1.2:  # Consider routes with ratio < 1.2 as "direct"
                    direct_routes += 1
        except nx.NetworkXNoPath:
            continue
    
    return {
        "average_directness": float(np.mean(directness_ratios)) if directness_ratios else 0.0,
        "direct_routes_ratio": float(direct_routes / len(pairs)) if pairs else 0.0
    }

def _calculate_block_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate block metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of block metrics
    """
    # Get all edges with length
    block_lengths = [float(d['length']) for _, _, d in G.edges(data=True) if 'length' in d]
    
    if not block_lengths:
        return {
            "average_length_meters": 0.0,
            "length_stats": {
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "std": 0.0
            },
            "size_distribution": []
        }
    
    # Calculate statistics
    avg_length = np.mean(block_lengths)
    
    # Create histogram for size distribution (10 bins)
    hist, bin_edges = np.histogram(block_lengths, bins=10)
    
    return {
        "average_length_meters": float(avg_length),
        "length_stats": {
            "min": float(min(block_lengths)),
            "max": float(max(block_lengths)),
            "median": float(np.median(block_lengths)),
            "std": float(np.std(block_lengths))
        },
        "size_distribution": {
            "counts": hist.tolist(),
            "bin_edges_meters": bin_edges.tolist()
        }
    }

def _find_orientation_peaks(histogram: List[float], threshold: float = 0.1) -> List[bool]:
    """Find peaks in orientation histogram.
    
    Args:
        histogram: List of orientation counts
        threshold: Peak detection threshold
        
    Returns:
        List of booleans indicating peak locations
    """
    peaks = []
    n = len(histogram)
    
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        
        is_peak = (
            histogram[i] > threshold and
            histogram[i] > histogram[prev_idx] and
            histogram[i] > histogram[next_idx]
        )
        peaks.append(is_peak)
        
    return peaks

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
        if degree > 2:  # Node is an intersection
            data = G.nodes[node]
            intersections.append((data['y'], data['x']))
    return intersections

def calculate_intersection_density(G: nx.MultiDiGraph) -> float:
    """Calculate intersection density (intersections per hectare).
    
    Args:
        G: NetworkX graph
        
    Returns:
        Intersection density (intersections/hectare)
    """
    if not G or G.number_of_nodes() == 0:
        return 0.0
    
    area_hectares = calculate_area_hectares(G) / 10000  # Convert m² to hectares
    intersections = len([n for n, d in G.degree() if d > 2])
    return intersections / area_hectares if area_hectares > 0 else 0

def calculate_network_density(G: nx.MultiDiGraph) -> float:
    """Calculate network density (total street length per hectare).
    
    Args:
        G: NetworkX graph
        
    Returns:
        Network density (kilometers/hectare)
    """
    if not G or G.number_of_nodes() == 0:
        return 0.0
    
    area_hectares = calculate_area_hectares(G) / 10000  # Convert m² to hectares
    total_length_km = sum(d["length"] for _, _, d in G.edges(data=True)) / 1000
    return total_length_km / area_hectares if area_hectares > 0 else 0

def calculate_network_features(G, feature_sets=None):
    """Calculate network features for machine learning.
    
    Args:
        G: NetworkX MultiDiGraph from OSMnx
        feature_sets: List of feature sets to calculate
        
    Returns:
        Dictionary of network features
    """
    if feature_sets is None:
        feature_sets = ['basic', 'centrality', 'efficiency']
        
    features = {}
    
    # Basic features
    if 'basic' in feature_sets:
        features.update({
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'density': nx.density(G)
        })
    
    # Centrality features
    if 'centrality' in feature_sets:
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            features.update({
                'avg_degree_centrality': np.mean(list(degree_centrality.values())),
                'avg_betweenness_centrality': np.mean(list(betweenness_centrality.values()))
            })
        except:
            features.update({
                'avg_degree_centrality': 0.0,
                'avg_betweenness_centrality': 0.0
            })
    
    # Efficiency features
    if 'efficiency' in feature_sets:
        try:
            features['global_efficiency'] = nx.global_efficiency(G)
        except:
            features['global_efficiency'] = 0.0
            
    return features

def download_network(
    latitude: float,
    longitude: float,
    radius_meters: int,
    network_type: str = "all"
) -> nx.MultiDiGraph:
    """Download street network data from OpenStreetMap.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Analysis radius in meters
        network_type: Type of network to download ('all', 'drive', 'walk', 'bike')
        
    Returns:
        NetworkX graph containing street network
        
    Raises:
        GeoFeatureKitError: If network download fails
    """
    try:
        # Configure OSMnx
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        # Create point and get network
        point = (latitude, longitude)
        G = ox.graph_from_point(point, dist=radius_meters, network_type=network_type)
        
        # Add metadata to graph
        G.graph['dist'] = radius_meters
        G.graph['center_lat'] = latitude
        G.graph['center_lon'] = longitude
        G.graph['network_type'] = network_type
        
        # Calculate area in hectares
        area_hectares = calculate_area_hectares(radius_meters)
        G.graph['area_hectares'] = area_hectares
        
        # Add edge bearings (must be done before projection)
        try:
            G = ox.bearing.add_edge_bearings(G)
        except Exception as e:
            # If bearing calculation fails, manually add bearings
            print(f"Warning: Bearing calculation failed ({e}), calculating manually...")
            for u, v, k, data in G.edges(keys=True, data=True):
                if 'geometry' in data and data['geometry'] is not None:
                    # Use geometry to calculate bearing
                    coords = list(data['geometry'].coords)
                    if len(coords) >= 2:
                        lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
                        lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
                        bearing = ox.bearing.calculate_bearing(lat1, lon1, lat2, lon2)
                        data['bearing'] = bearing
                    else:
                        data['bearing'] = 0.0
                else:
                    # Use node coordinates to calculate bearing
                    lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
                    lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
                    bearing = ox.bearing.calculate_bearing(lat1, lon1, lat2, lon2)
                    data['bearing'] = bearing
        
        # Project to UTM
        G = ox.project_graph(G)
        
        # Edge lengths are automatically added by OSMnx during graph creation
        
        return G
        
    except Exception as e:
        raise GeoFeatureKitError(f"Failed to download network: {str(e)}")

def _calculate_density_metrics(
    G: nx.MultiDiGraph,
    area_hectares: float,
    total_length_km: float
) -> Dict[str, float]:
    """Calculate network density metrics.
    
    Args:
        G: NetworkX graph
        area_hectares: Area in hectares
        total_length_km: Total street length in kilometers
        
    Returns:
        Dictionary of density metrics (all per hectare)
    """
    # Calculate density metrics
    density_metrics = {
        "intersection_density": round_float(
            len([n for n, d in G.degree() if d > 2]) / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
        ),
        "street_density_km": round_float(
            total_length_km / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
        ),
        "node_density": round_float(
            G.number_of_nodes() / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
        )
    }
    
    return density_metrics 