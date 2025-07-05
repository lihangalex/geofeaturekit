"""Advanced network analysis utilities for pattern recognition and ML features."""

import numpy as np
import networkx as nx
import osmnx as ox
from node2vec import Node2Vec
from scipy import sparse
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from sklearn.decomposition import PCA
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import geopandas as gpd
from .area import calculate_area_hectares
from ..utils.progress import create_progress_bar, log_analysis_start, log_analysis_complete, log_error
from .formatting import (
    round_float,
    LENGTH_DECIMALS,
    AREA_DECIMALS,
    DENSITY_DECIMALS,
    RATIO_DECIMALS,
    ANGLE_DECIMALS
)

def calculate_advanced_metrics(G: nx.MultiDiGraph, area_hectares: float) -> Dict[str, Any]:
    """Calculate advanced network metrics.
    
    Args:
        G: NetworkX graph
        area_hectares: Area in hectares
        
    Returns:
        Dictionary containing advanced metrics
    """
    if not G or G.number_of_nodes() == 0:
        return {}

    # Calculate total length in kilometers
    total_length_km = sum(d["length"] for _, _, d in G.edges(data=True)) / 1000

    # Calculate density metrics
    density_metrics = {
        "intersections_per_hectare": round_float(
            len([n for n, d in G.degree() if d > 2]) / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
        ),
        "street_km_per_hectare": round_float(
            total_length_km / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
        ),
        "nodes_per_hectare": round_float(
            G.number_of_nodes() / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
        ),
        "units": "per_hectare"
    }
    
    # Calculate other metrics
    G_undirected = G.to_undirected()
    avg_degree = float(np.mean([d for _, d in G_undirected.degree()]))
    
    return {
        "density": density_metrics,
        "average_degree": round_float(avg_degree, RATIO_DECIMALS)
    }

def compute_urban_form_metrics(G: nx.MultiDiGraph, area_sqkm: float) -> Dict[str, Any]:
    """Compute metrics describing urban form and structure.
    
    Args:
        G: NetworkX graph
        area_sqkm: Area in square kilometers
        
    Returns:
        Dictionary of urban form metrics
    """
    # Calculate network efficiency
    efficiency = compute_network_efficiency(G)
    
    # Calculate network complexity
    complexity = compute_network_complexity(G)
    
    return {
        "efficiency": efficiency,
        "complexity": complexity
    }

def compute_network_patterns(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Analyze network patterns and structure.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of pattern metrics
    """
    # Calculate grid metrics
    grid_metrics = compute_grid_metrics(G)
    
    # Calculate centrality patterns
    centrality = compute_centrality_patterns(G)
    
    # Calculate network embeddings
    embeddings = compute_network_embeddings(G, dimensions=64)
    
    return {
        "grid_metrics": grid_metrics,
        "centrality_patterns": centrality,
        "embeddings": embeddings
    }

def compute_accessibility_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Compute accessibility and reachability metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of accessibility metrics
    """
    # Convert to undirected for some calculations
    G_undirected = G.to_undirected()
    
    # Calculate reach metrics
    reach = compute_reach_metrics(G_undirected)
    
    # Calculate betweenness centrality (for major routes)
    betweenness = nx.betweenness_centrality(G_undirected)
    major_routes = identify_major_routes(G_undirected, betweenness)
    
    return {
        "reach_metrics": reach,
        "major_routes": major_routes
    }

def compute_network_efficiency(G: nx.Graph) -> Dict[str, float]:
    """Calculate network efficiency metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of efficiency metrics
    """
    # Sample nodes if graph is large
    nodes = list(G.nodes())
    if len(nodes) > 100:
        np.random.seed(42)
        nodes = np.random.choice(nodes, 100, replace=False)
    
    # Calculate average shortest path length
    path_lengths = []
    for i, start in enumerate(nodes):
        for end in nodes[i+1:]:
            try:
                length = nx.shortest_path_length(G, start, end, weight='length')
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                continue
    
    if path_lengths:
        avg_path = np.mean(path_lengths)
        path_std = np.std(path_lengths)
    else:
        avg_path = 0
        path_std = 0
    
    return {
        "average_path_length_meters": float(avg_path),
        "path_length_std_meters": float(path_std),
        "network_diameter_meters": float(max(path_lengths)) if path_lengths else 0
    }

def compute_network_complexity(G: nx.Graph) -> Dict[str, float]:
    """Calculate network complexity metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of complexity metrics
    """
    # Calculate basic graph metrics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # Calculate cyclomatic number (number of fundamental cycles)
    components = nx.number_connected_components(G)
    cyclomatic = n_edges - n_nodes + components
    
    # Calculate edge density
    max_edges = n_nodes * (n_nodes - 1) / 2
    edge_density = n_edges / max_edges if max_edges > 0 else 0
    
    return {
        "cyclomatic_number": int(cyclomatic),
        "edge_density": float(edge_density),
        "average_node_degree": float(2 * n_edges / n_nodes) if n_nodes > 0 else 0
    }

def compute_grid_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Analyze grid-like patterns in the network.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of grid metrics
    """
    # Calculate street orientations
    orientations = []
    for _, _, data in G.edges(data=True):
        if 'bearing' in data:
            # Normalize to 0-180 degrees
            orientation = data['bearing'] % 180
            orientations.append(orientation)
    
    if not orientations:
        return {
            "grid_score": 0.0,
            "dominant_angles": [],
            "regularity_score": 0.0
        }
    
    # Calculate orientation entropy
    hist, _ = np.histogram(orientations, bins=18, range=(0, 180))
    hist = hist / len(orientations)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Find dominant angles
    peaks = find_orientation_peaks(hist)
    dominant_angles = [i * 10 for i, is_peak in enumerate(peaks) if is_peak]
    
    # Calculate grid score based on perpendicular streets
    grid_score = calculate_grid_score(dominant_angles)
    
    return {
        "grid_score": float(grid_score),
        "dominant_angles": dominant_angles,
        "regularity_score": float(1 - entropy/4.17)  # Normalize by max entropy
    }

def compute_centrality_patterns(G: nx.Graph) -> Dict[str, Any]:
    """Analyze centrality patterns in the network.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of centrality patterns
    """
    # Calculate different centrality measures
    degree_cent = nx.degree_centrality(G)
    close_cent = nx.closeness_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    
    # Calculate statistics for each measure
    measures = {
        "degree": degree_cent,
        "closeness": close_cent,
        "betweenness": between_cent
    }
    
    patterns = {}
    for name, measure in measures.items():
        values = list(measure.values())
        patterns[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(max(values)),
            "distribution": np.histogram(values, bins=10)[0].tolist()
        }
    
    return patterns

def compute_reach_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Calculate reach metrics for different distances.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of reach metrics
    """
    distances = [400, 800, 1200]  # Common walking distances in meters
    reach_metrics = {}
    
    for dist in distances:
        reachable_length = 0
        reachable_nodes = 0
        
        # Sample nodes if graph is large
        nodes = list(G.nodes())
        if len(nodes) > 50:
            np.random.seed(42)
            nodes = np.random.choice(nodes, 50, replace=False)
        
        for node in nodes:
            # Get all nodes within distance
            length = nx.single_source_dijkstra_path_length(G, node, cutoff=dist, weight='length')
            reachable_length += sum(length.values())
            reachable_nodes += len(length)
        
        if nodes:
            avg_length = reachable_length / len(nodes)
            avg_nodes = reachable_nodes / len(nodes)
        else:
            avg_length = 0
            avg_nodes = 0
        
        reach_metrics[f"{dist}m"] = {
            "average_reachable_length_meters": float(avg_length),
            "average_reachable_nodes": float(avg_nodes)
        }
    
    return reach_metrics

def identify_major_routes(G: nx.Graph, betweenness: Dict) -> List[Dict[str, Any]]:
    """Identify major routes in the network.
    
    Args:
        G: NetworkX graph
        betweenness: Betweenness centrality values
        
    Returns:
        List of major routes
    """
    # Get high betweenness nodes
    threshold = np.percentile(list(betweenness.values()), 90)
    major_nodes = [node for node, cent in betweenness.items() if cent > threshold]
    
    # Find paths between major nodes
    major_routes = []
    for i, start in enumerate(major_nodes):
        for end in major_nodes[i+1:]:
            try:
                path = nx.shortest_path(G, start, end, weight='length')
                length = sum(G[path[i]][path[i+1]].get('length', 0) 
                           for i in range(len(path)-1))
                
                major_routes.append({
                    "start_node": str(start),
                    "end_node": str(end),
                    "path_nodes": [str(n) for n in path],
                    "length_meters": float(length),
                    "importance_score": float(betweenness[start] * betweenness[end])
                })
            except nx.NetworkXNoPath:
                continue
    
    # Sort by importance
    major_routes.sort(key=lambda x: x['importance_score'], reverse=True)
    return major_routes[:10]  # Return top 10 routes

def find_orientation_peaks(histogram: np.ndarray, threshold: float = 0.1) -> List[bool]:
    """Find peaks in orientation histogram.
    
    Args:
        histogram: Normalized histogram of orientations
        threshold: Minimum relative height for a peak
        
    Returns:
        List of booleans indicating which bins are peaks
    """
    peaks = []
    for i in range(len(histogram)):
        prev_val = histogram[(i - 1) % len(histogram)]
        curr_val = histogram[i]
        next_val = histogram[(i + 1) % len(histogram)]
        
        is_peak = (curr_val > threshold and
                  curr_val > prev_val and
                  curr_val > next_val)
        peaks.append(is_peak)
    
    return peaks

def calculate_grid_score(angles: List[int]) -> float:
    """Calculate grid score based on dominant angles.
    
    Args:
        angles: List of dominant angles in degrees
        
    Returns:
        Grid score between 0 and 1
    """
    if not angles:
        return 0.0
    
    # Check for perpendicular pairs
    perpendicular_pairs = 0
    total_pairs = 0
    
    for i, angle1 in enumerate(angles):
        for angle2 in angles[i+1:]:
            total_pairs += 1
            if abs((angle1 - angle2) % 90) < 10:  # Allow 10° tolerance
                perpendicular_pairs += 1
    
    return perpendicular_pairs / total_pairs if total_pairs > 0 else 0.0

def compute_centrality_measures(G: nx.MultiDiGraph) -> dict:
    """Compute various centrality measures for the network."""
    # Convert to undirected for certain metrics
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
    
    # Calculate averages
    return {
        "avg_betweenness": float(np.mean(list(betweenness.values()))),
        "avg_closeness": float(np.mean(list(closeness.values()))),
        "avg_degree": float(np.mean(list(degree.values())))
    }

def compute_road_distribution(G: nx.MultiDiGraph) -> dict:
    """Compute distribution of road types."""
    road_types = {}
    total_length = 0
    
    for _, _, data in G.edges(data=True):
        road_type = data.get('highway', 'unknown')
        length = float(data.get('length', 0))
        total_length += length
        
        if isinstance(road_type, list):
            road_type = road_type[0]
        
        road_types[road_type] = road_types.get(road_type, 0) + length
    
    # Convert to percentages
    if total_length > 0:
        distribution = {k: float(v/total_length) for k, v in road_types.items()}
    else:
        distribution = {}
    
    return distribution

def compute_network_embeddings(
    G: nx.MultiDiGraph,
    dimensions: int = 128,
    reduce_dims: Optional[int] = None,
    num_walks: int = 200,
    walk_length: int = 30
) -> List[float]:
    """Compute Node2Vec embeddings for the network.
    
    Args:
        G: Input graph
        dimensions: Number of dimensions for embeddings
        reduce_dims: If set, reduce embeddings to this many dimensions using PCA
        num_walks: Number of random walks per node
        walk_length: Length of each random walk
        
    Returns:
        List of embedding values
    """
    # Convert to undirected for Node2Vec
    G_undirected = G.to_undirected()
    
    # Convert node IDs to strings and handle array values
    G_clean = nx.Graph()
    node_mapping = {}  # Keep track of original to clean node IDs
    
    for node in G_undirected.nodes():
        # Convert array node IDs to strings
        if isinstance(node, (list, np.ndarray)):
            clean_id = str(list(node))  # Convert array to string representation
        else:
            clean_id = str(node)
        node_mapping[node] = clean_id
        G_clean.add_node(clean_id)
    
    # Add edges using clean node IDs
    for u, v in G_undirected.edges():
        G_clean.add_edge(node_mapping[u], node_mapping[v])
    
    # Initialize Node2Vec
    try:
        node2vec = Node2Vec(
            G_clean,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=1  # Single worker for consistency
        )
        
        # Train the model
        model = node2vec.fit(window=10, min_count=1)
        
        # Get embeddings for all nodes
        node_embeddings = []
        for node in G_clean.nodes():
            try:
                node_embeddings.append(model.wv[node])
            except KeyError:
                continue
        
        # Return average embedding across all nodes
        if node_embeddings:
            # Stack embeddings into a matrix for reduction
            embeddings_matrix = np.stack(node_embeddings)
            
            # Reduce dimensionality if requested
            if reduce_dims is not None and reduce_dims < dimensions:
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
    except Exception as e:
        print(f"  Warning: Error computing network embeddings: {str(e)}")
    
    # Return zero vector of appropriate size if computation fails
    return [0.0] * (reduce_dims if reduce_dims is not None else dimensions)

def compute_advanced_stats(G: nx.MultiDiGraph) -> dict:
    """Compute all advanced network statistics."""
    # Calculate area in hectares
    radius_meters = G.graph['dist']
    area_hectares = calculate_area_hectares(radius_meters)
    
    # Get advanced metrics
    centrality = compute_centrality_measures(G)
    road_dist = compute_road_distribution(G)
    embeddings = compute_network_embeddings(G)
    
    # Calculate additional metrics
    intersections = len([n for n, d in G.degree() if d > 2])
    intersection_density = intersections / area_hectares if area_hectares > 0 else 0
    
    # Average block length
    block_lengths = [float(d['length']) for _, _, d in G.edges(data=True) if 'length' in d]
    avg_block_length = np.mean(block_lengths) if block_lengths else 0
    
    return {
        "centrality": centrality,
        "road_distribution": road_dist,
        "density": {
            "intersections_per_hectare": round_float(intersection_density, DENSITY_DECIMALS),
            "average_block_length_meters": round_float(avg_block_length, LENGTH_DECIMALS),
            "units": "per_hectare"
        },
        "network_embeddings": embeddings
    }

def compute_space_syntax_metrics(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Compute space syntax metrics for the street network."""
    G_undirected = G.to_undirected()
    
    # Calculate integration (normalized closeness centrality)
    closeness = nx.closeness_centrality(G_undirected)
    
    # Calculate choice (normalized betweenness centrality)
    betweenness = nx.betweenness_centrality(G_undirected)
    
    # Calculate connectivity (degree centrality)
    degree = nx.degree_centrality(G_undirected)
    
    return {
        "global_integration": float(np.mean(list(closeness.values()))),
        "global_choice": float(np.mean(list(betweenness.values()))),
        "connectivity": float(np.mean(list(degree.values()))),
        "integration_std": float(np.std(list(closeness.values()))),
        "choice_std": float(np.std(list(betweenness.values())))
    }

def compute_orientation_entropy(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Compute street network orientation entropy and patterns."""
    bearings = []
    
    for _, _, data in G.edges(data=True):
        # Get the bearing from the geometry if available
        if 'bearing' in data:
            # Normalize bearing to 0-180
            bearing = data['bearing'] % 180
            bearings.append(bearing)
    
    if not bearings:
        return {
            "orientation_entropy": 0.0,
            "grid_pattern_strength": 0.0,
            "dominant_orientation": 0.0
        }
    
    # Calculate entropy
    hist, _ = np.histogram(bearings, bins=18, range=(0, 180))  # 10° bins
    hist = hist / len(bearings)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Detect grid patterns (0°, 90°, 45°, 135°)
    grid_angles = [0, 45, 90, 135]
    grid_strength = sum(sum(1 for x in bearings if abs((x - angle) % 180) < 5)
                       for angle in grid_angles) / len(bearings)
    
    return {
        "orientation_entropy": float(entropy),
        "grid_pattern_strength": float(grid_strength),
        "dominant_orientation": float(np.median(bearings))
    }

def compute_morphological_metrics(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Compute morphological metrics of the street network."""
    # Get network bounds
    radius_meters = G.graph['dist']
    area_sqkm = calculate_area_hectares(radius_meters)
    
    # Basic metrics
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    total_length = sum(d['length'] for _, _, d in G.edges(data=True))
    
    # Derived metrics
    connectivity_index = edge_count / node_count if node_count > 0 else 0
    network_density = total_length / area_sqkm if area_sqkm > 0 else 0
    
    # Calculate organic ratio (more organic = higher value)
    straight_edges = sum(1 for _, _, d in G.edges(data=True) 
                        if 'geometry' not in d)
    organic_ratio = 1 - (straight_edges / edge_count if edge_count > 0 else 0)
    
    return {
        "connectivity_index": round_float(connectivity_index, 2),
        "network_density_meters_per_sqkm": round_float(network_density, 1),
        "organic_ratio": round_float(organic_ratio, 2)
    }

def check_road_type(highway: Any, valid_types: List[str]) -> bool:
    """Check if a road type matches any of the valid types.
    
    Args:
        highway: The road type to check (can be scalar or array-like)
        valid_types: List of valid road types
        
    Returns:
        bool: True if there's a match, False otherwise
    """
    if highway is None:
        return False
        
    # Convert valid types to set for faster lookup
    valid_set = set(valid_types)
    
    try:
        # Handle list-like objects without converting to numpy array
        if hasattr(highway, '__iter__') and not isinstance(highway, (str, bytes)):
            # Iterate directly over the values
            for val in highway:
                if pd.notna(val) and str(val) in valid_set:
                    return True
            return False
        else:
            # Handle scalar value
            return pd.notna(highway) and str(highway) in valid_set
    except Exception as e:
        print(f"  Warning: Error checking road type: {str(e)}")
        return False

def compute_hierarchical_metrics(G: nx.MultiDiGraph) -> Dict[str, Dict[str, float]]:
    """Compute hierarchical street network metrics."""
    road_hierarchy = {
        'primary': ['motorway', 'trunk', 'primary'],
        'secondary': ['secondary', 'tertiary'],
        'local': ['residential', 'living_street', 'unclassified']
    }
    
    metrics = {}
    total_length = sum(d['length'] for _, _, _, d in G.edges(data=True, keys=True))
    
    for level, road_types in road_hierarchy.items():
        # Filter edges by road type
        level_edges = []
        level_nodes = set()
        for u, v, k, d in G.edges(data=True, keys=True):
            highway = d.get('highway', '')
            if check_road_type(highway, road_types):
                level_edges.append((u, v))
                level_nodes.add(u)
                level_nodes.add(v)
        
        if level_edges:
            # Calculate length for this level
            level_length = sum(d['length'] for _, _, _, d in G.edges(data=True, keys=True)
                             if check_road_type(d.get('highway', ''), road_types))
            level_proportion = level_length / total_length if total_length > 0 else 0
            
            # Create subgraph for this level
            level_graph = G.subgraph(level_nodes).copy()
            
            # Calculate average betweenness for this level
            if level_graph.number_of_nodes() > 1:
                try:
                    betweenness = nx.betweenness_centrality(level_graph.to_undirected())
                    avg_betweenness = np.mean(list(betweenness.values()))
                except:
                    avg_betweenness = 0
            else:
                avg_betweenness = 0
                
            metrics[level] = {
                "proportion": float(level_proportion),
                "avg_betweenness": float(avg_betweenness)
            }
        else:
            metrics[level] = {
                "proportion": 0.0,
                "avg_betweenness": 0.0
            }
    
    return metrics

def compute_spectral_features(G: nx.MultiDiGraph) -> List[float]:
    """Compute spectral features of the network."""
    try:
        # Convert to undirected and get largest connected component
        G_undirected = G.to_undirected()
        if not nx.is_connected(G_undirected):
            G_undirected = G_undirected.subgraph(max(nx.connected_components(G_undirected)))
        
        # Get adjacency matrix
        A = nx.adjacency_matrix(G_undirected)
        
        # Compute normalized Laplacian
        n_nodes = A.shape[0]
        
        # Handle array values safely
        row_sums = np.asarray(A.sum(axis=1)).flatten()  # Convert to flat array
        d_values = np.zeros(n_nodes)
        for i in range(n_nodes):
            if row_sums[i] > 0:  # Avoid division by zero
                d_values[i] = 1.0 / np.sqrt(row_sums[i])
        
        D = sparse.diags(d_values)
        L = sparse.eye(n_nodes) - D @ A @ D
        
        # Compute eigenvalues (use only first 10)
        eigenvals = sparse.linalg.eigsh(L, k=min(10, n_nodes-1), which='SM', 
                                      return_eigenvectors=False)
        
        # Pad with zeros if needed
        spectral_features = list(eigenvals)
        spectral_features.extend([0.0] * (10 - len(spectral_features)))
        
        return [float(x) for x in spectral_features]
    except Exception as e:
        print(f"  Warning: Error computing spectral features: {str(e)}")
        return [0.0] * 10

def download_land_use(
    latitude: float,
    longitude: float,
    radius_meters: int,
    custom_tags: Optional[Dict[str, Any]] = None
) -> gpd.GeoDataFrame:
    """Download land use data from OpenStreetMap.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Analysis radius in meters
        custom_tags: Custom OSM tags to filter land use
        
    Returns:
        GeoDataFrame containing land use polygons
    """
    # Configure OSMnx
    ox.settings.use_cache = True
    ox.settings.log_console = False
    
    # Default land use tags
    tags = {
        'landuse': True,
        'leisure': True,
        'natural': True,
        'building': True
    }
    
    # Add custom tags if provided
    if custom_tags:
        tags.update(custom_tags)
    
    # Download features
    features = ox.features_from_point(
        (latitude, longitude),
        tags=tags,
        dist=radius_meters
    )
    
    # Filter to just polygons
    polygons = features[
        features.geometry.type.isin(['Polygon', 'MultiPolygon'])
    ].copy()
    
    # Classify land use
    def classify_land_use(row):
        if 'landuse' in row and row['landuse']:
            if row['landuse'] in ['residential', 'apartments']:
                return 'residential'
            elif row['landuse'] in ['commercial', 'retail']:
                return 'commercial'
            elif row['landuse'] in ['mixed']:
                return 'mixed_use'
        
        if 'leisure' in row and row['leisure'] in ['park', 'garden']:
            return 'open_space'
        if 'natural' in row and row['natural'] in ['wood', 'grassland']:
            return 'open_space'
            
        return None
    
    polygons['landuse'] = polygons.apply(classify_land_use, axis=1)
    return polygons[polygons['landuse'].notna()].copy()

def calculate_intersection_density(G: nx.MultiDiGraph) -> float:
    """Calculate intersection density (intersections per hectare).
    
    Args:
        G: NetworkX graph
        
    Returns:
        Intersection density (intersections/hectare)
    """
    if not G or G.number_of_nodes() == 0:
        return 0.0
    
    area_hectares = calculate_area_hectares(G.graph['dist'])
    intersections = len([n for n, d in G.degree() if d > 2])
    return round_float(intersections / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS)

def calculate_network_density(G: nx.MultiDiGraph) -> float:
    """Calculate network density (total street length per hectare).
    
    Args:
        G: NetworkX graph
        
    Returns:
        Network density (kilometers/hectare)
    """
    if not G or G.number_of_nodes() == 0:
        return 0.0
    
    area_hectares = calculate_area_hectares(G.graph['dist'])
    total_length_km = sum(d["length"] for _, _, d in G.edges(data=True)) / 1000
    return round_float(total_length_km / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS)

def calculate_intersection_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate intersection-related metrics.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary containing intersection metrics
    """
    if not G or G.number_of_nodes() == 0:
        return {}

    # Calculate area in hectares
    area_hectares = calculate_area_hectares(G.graph['dist'])
    
    # Count intersections
    intersections = len([n for n, d in G.degree() if d > 2])
    
    # Calculate intersection density
    intersection_density = intersections / area_hectares if area_hectares > 0 else 0

    return {
        "density": {
            "intersections_per_hectare": round_float(intersection_density, DENSITY_DECIMALS),
            "units": "per_hectare"
        },
        "total_intersections": intersections,
        "area_hectares": round_float(area_hectares, AREA_DECIMALS)
    }

def calculate_network_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate network-related metrics.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary containing network metrics
    """
    if not G or G.number_of_nodes() == 0:
        return {}

    # Calculate area in hectares
    area_hectares = calculate_area_hectares(G.graph['dist'])
    
    # Calculate total length
    total_length = sum(d["length"] for _, _, d in G.edges(data=True))
    total_length_km = total_length / 1000
    
    # Calculate network density
    network_density = total_length_km / area_hectares if area_hectares > 0 else 0

    return {
        "length": {
            "total_length_meters": round_float(total_length, LENGTH_DECIMALS),
            "total_length_km": round_float(total_length_km, LENGTH_DECIMALS)
        },
        "density": {
            "street_km_per_hectare": round_float(network_density, DENSITY_DECIMALS),
            "units": "per_hectare"
        },
        "area_hectares": round_float(area_hectares, AREA_DECIMALS)
    } 