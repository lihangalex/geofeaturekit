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
    radius_meters: int,
    feature_sets: Optional[Set[str]] = None,
    compute_embeddings: bool = False,
    embedding_config: Optional[Dict] = None,
    shared_embeddings: Optional[Dict[str, List[float]]] = None,
    location_key: Optional[str] = None
) -> Dict[str, Any]:
    """Get street network statistics for a location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Analysis radius in meters
        feature_sets: Set of feature types to compute
            Options: ['basic', 'urban', 'spectral']
            If None, defaults to ['basic']
        compute_embeddings: Whether to compute contextual embeddings
        embedding_config: Configuration for embedding computation
            - dimensions: Number of dimensions
            - reduce_dims: Optional dimensionality reduction
            - num_walks: Number of random walks
            - walk_length: Length of random walks
        shared_embeddings: Optional pre-computed embeddings for shared graph
        location_key: Optional key to look up embeddings in shared_embeddings
        
    Returns:
        Dictionary containing requested feature sets
    """
    # Default to basic features
    feature_sets = feature_sets or {'basic'}
    
    # Create point and get network
    point = (latitude, longitude)
    G = ox.graph_from_point(point, dist=radius_meters, network_type='all')
    
    # Add center point and radius to graph metadata for area calculations
    G.graph['center_lat'] = latitude
    G.graph['center_lon'] = longitude
    G.graph['dist'] = radius_meters
    
    # Convert to undirected graph for analysis
    G_undirected = G.to_undirected()
    
    # Initialize results
    results = {}
    
    # Basic features (always computed)
    if 'basic' in feature_sets:
        total_length = sum(d['length'] for u, v, d in G.edges(data=True))
        intersections = len([node for node, degree in G_undirected.degree() if degree > 2])
        street_segments = G.number_of_edges()
        
        results['basic'] = {
            'total_length': total_length,
            'intersections': intersections,
            'segments': street_segments,
            'centrality': compute_basic_centrality(G),
            'road_distribution': compute_road_distribution(G),
            'density': compute_density_metrics(G)
        }
    
    # Urban features
    if 'urban' in feature_sets:
        results['urban'] = {
            'space_syntax': compute_space_syntax_metrics(G),
            'orientation': compute_orientation_entropy(G),
            'morphology': compute_morphological_metrics(G),
            'hierarchy': compute_hierarchical_metrics(G)
        }
    
    # Spectral features
    if 'spectral' in feature_sets:
        results['spectral'] = {
            'features': compute_spectral_features(G)
        }
    
    # Contextual embeddings (if requested)
    if compute_embeddings:
        if shared_embeddings and location_key:
            # Use pre-computed embeddings
            results['embeddings'] = shared_embeddings[location_key]
        elif embedding_config is None:
            embedding_config = {
                'dimensions': 128,
                'reduce_dims': None,
                'num_walks': 200,
                'walk_length': 30
            }
            results['embeddings'] = compute_contextual_embeddings(
                G,
                dimensions=embedding_config['dimensions'],
                reduce_dims=embedding_config['reduce_dims'],
                num_walks=embedding_config['num_walks'],
                walk_length=embedding_config['walk_length']
            )
    
    return results

def compute_basic_centrality(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Compute basic centrality measures."""
    G_undirected = G.to_undirected()
    
    # Sample nodes if network is large
    if len(G) > 1000:
        sampled_nodes = np.random.choice(list(G.nodes()), 1000, replace=False)
        G_sample = G.subgraph(sampled_nodes)
        G_undirected_sample = G_undirected.subgraph(sampled_nodes)
    else:
        G_sample = G
        G_undirected_sample = G_undirected
    
    # Calculate centrality measures
    betweenness = nx.betweenness_centrality(G_sample)
    closeness = nx.closeness_centrality(G_undirected_sample)
    degree = dict(G.degree())
    
    return {
        "avg_betweenness": float(np.mean(list(betweenness.values()))),
        "avg_closeness": float(np.mean(list(closeness.values()))),
        "avg_degree": float(np.mean(list(degree.values())))
    }

def compute_road_distribution(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Compute distribution of road types."""
    road_types = {}
    total_length = 0
    
    for _, _, data in G.edges(data=True):
        road_type = process_road_type(data.get('highway'))
        length = float(data.get('length', 0))
        total_length += length
        
        road_types[road_type] = road_types.get(road_type, 0) + length
    
    # Convert to percentages
    if total_length > 0:
        distribution = {k: float(v/total_length) for k, v in road_types.items()}
    else:
        distribution = {}
    
    return distribution

def compute_density_metrics(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Compute network density metrics."""
    # Calculate area (approximate)
    bounds = ox.utils_geo.bbox_from_point(
        (G.graph['center_lat'], G.graph['center_lon']),
        dist=G.graph['dist']
    )
    area_sqkm = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 111) ** 2
    
    # Calculate metrics
    total_length = sum(d['length'] for _, _, d in G.edges(data=True))
    intersections = len([n for n, d in G.degree() if d > 2])
    
    return {
        "network_density": float(total_length / area_sqkm) if area_sqkm > 0 else 0,
        "intersection_density": float(intersections / area_sqkm) if area_sqkm > 0 else 0
    }

def compute_contextual_embeddings(
    G: nx.MultiDiGraph,
    dimensions: int = 128,
    reduce_dims: Optional[int] = None,
    num_walks: int = 200,
    walk_length: int = 30
) -> List[float]:
    """Compute Node2Vec embeddings for the network.
    
    Args:
        G: NetworkX graph
        dimensions: Number of dimensions for embeddings
        reduce_dims: If set, reduce embeddings to this many dimensions
        num_walks: Number of random walks per node
        walk_length: Length of each random walk
        
    Returns:
        List of embedding values (averaged across nodes)
    """
    # Convert to undirected for Node2Vec
    G_undirected = G.to_undirected()
    
    # Initialize Node2Vec
    node2vec = Node2Vec(
        G_undirected,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1  # Single worker for consistency
    )
    
    # Train the model
    model = node2vec.fit(window=10, min_count=1)
    
    # Get embeddings for all nodes
    node_embeddings = []
    for node in G_undirected.nodes():
        try:
            node_embeddings.append(model.wv[str(node)])
        except KeyError:
            continue
    
    # Return average embedding across all nodes
    if node_embeddings:
        # Stack embeddings into a matrix for reduction
        embeddings_matrix = np.stack(node_embeddings)
        
        # Reduce dimensionality if requested
        if reduce_dims is not None and reduce_dims < dimensions:
            # First reduce individual node embeddings
            pca = PCA(n_components=min(reduce_dims, embeddings_matrix.shape[0]))
            reduced_embeddings = pca.fit_transform(embeddings_matrix)
            
            # Then average the reduced embeddings
            avg_embedding = np.mean(reduced_embeddings, axis=0)
            
            # Pad with zeros if needed
            if len(avg_embedding) < reduce_dims:
                avg_embedding = np.pad(avg_embedding, (0, reduce_dims - len(avg_embedding)))
            
            return avg_embedding.tolist()
        
        # If no reduction needed, just average the original embeddings
        avg_embedding = np.mean(embeddings_matrix, axis=0)
        return avg_embedding.tolist()
    
    # Return zero vector of appropriate size if no embeddings
    return [0.0] * (reduce_dims if reduce_dims is not None else dimensions)

def compute_contextual_embeddings_batch(
    locations: List[Dict[str, float]],
    radius_meters: int,
    dimensions: int = 128,
    reduce_dims: Optional[int] = None,
    num_walks: int = 200,
    walk_length: int = 30
) -> Dict[str, List[float]]:
    """Compute contextual embeddings for multiple locations using a shared graph.
    
    Args:
        locations: List of location dictionaries with 'latitude' and 'longitude'
        radius_meters: Analysis radius in meters
        dimensions: Number of dimensions for embeddings
        reduce_dims: If set, reduce embeddings to this many dimensions
        num_walks: Number of random walks per node
        walk_length: Length of each random walk
        
    Returns:
        Dictionary mapping location strings to their embeddings
    """
    # Create a combined graph
    G_combined = nx.MultiDiGraph()
    location_nodes = {}
    
    log_analysis_start(len(locations))
    
    # Build combined graph
    for loc in create_progress_bar(locations, desc="Building combined graph"):
        lat, lon = loc['latitude'], loc['longitude']
        point = (lat, lon)
        G = ox.graph_from_point(point, dist=radius_meters, network_type='all')
        
        # Add center node to track location
        center_node = f"{lat},{lon}"  # Use string format
        G_combined.add_node(center_node, pos=(lat, lon))
        location_nodes[center_node] = G.nodes()
        
        # Add all nodes and edges from this location's graph
        G_combined.add_nodes_from(G.nodes(data=True))
        G_combined.add_edges_from(G.edges(data=True))
    
    # Convert to undirected for Node2Vec
    G_undirected = G_combined.to_undirected()
    
    # Initialize Node2Vec
    node2vec = Node2Vec(
        G_undirected,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1  # Single worker for consistency
    )
    
    # Train the model
    print("Computing embeddings...")
    model = node2vec.fit(window=10, min_count=1)
    
    # Get embeddings for each location's subgraph
    embeddings = {}
    for loc in create_progress_bar(locations, desc="Extracting location embeddings"):
        lat, lon = loc['latitude'], loc['longitude']
        center_node = f"{lat},{lon}"  # Use string format
        
        # Get embeddings for nodes in this location's subgraph
        node_embeddings = []
        for node in location_nodes[center_node]:
            try:
                node_embeddings.append(model.wv[str(node)])
            except KeyError:
                continue
        
        # Stack embeddings into a matrix for reduction
        if node_embeddings:
            embeddings_matrix = np.stack(node_embeddings)
            
            # Reduce dimensionality if requested
            if reduce_dims is not None and reduce_dims < dimensions:
                # First reduce individual node embeddings
                pca = PCA(n_components=min(reduce_dims, embeddings_matrix.shape[0]))
                reduced_embeddings = pca.fit_transform(embeddings_matrix)
                
                # Then average the reduced embeddings
                avg_embedding = np.mean(reduced_embeddings, axis=0)
                
                # Pad with zeros if needed
                if len(avg_embedding) < reduce_dims:
                    avg_embedding = np.pad(avg_embedding, (0, reduce_dims - len(avg_embedding)))
                
                embeddings[center_node] = avg_embedding.tolist()
            else:
                # If no reduction needed, just average the original embeddings
                avg_embedding = np.mean(embeddings_matrix, axis=0)
                embeddings[center_node] = avg_embedding.tolist()
        else:
            # Return zero vector if no embeddings
            embeddings[center_node] = [0.0] * (reduce_dims if reduce_dims is not None else dimensions)
    
    log_analysis_complete(len(locations), len(embeddings))
    return embeddings 

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