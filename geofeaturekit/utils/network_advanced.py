"""Advanced network analysis utilities."""

import numpy as np
import networkx as nx
import osmnx as ox
from node2vec import Node2Vec
from scipy import sparse
from typing import Dict, List, Any, Optional
import pandas as pd

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
                from sklearn.decomposition import PCA
                
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
    except Exception as e:
        print(f"  Warning: Error computing network embeddings: {str(e)}")
    
    # Return zero vector of appropriate size if computation fails
    return [0.0] * (reduce_dims if reduce_dims is not None else dimensions)

def compute_advanced_stats(G: nx.MultiDiGraph) -> dict:
    """Compute all advanced network statistics."""
    # Basic area calculation (approximate)
    bounds = ox.utils_geo.bbox_from_point(
        (G.graph['center_lat'], G.graph['center_lon']),
        dist=G.graph['dist']
    )
    area_sqkm = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 111) ** 2
    
    # Get advanced metrics
    centrality = compute_centrality_measures(G)
    road_dist = compute_road_distribution(G)
    embeddings = compute_network_embeddings(G)
    
    # Calculate additional metrics
    intersections = len([n for n, d in G.degree() if d > 2])
    intersection_density = intersections / area_sqkm if area_sqkm > 0 else 0
    
    # Average block length
    block_lengths = [float(d['length']) for _, _, d in G.edges(data=True) if 'length' in d]
    avg_block_length = np.mean(block_lengths) if block_lengths else 0
    
    return {
        "centrality": centrality,
        "road_distribution": road_dist,
        "density": {
            "intersection_density": float(intersection_density),
            "avg_block_length": float(avg_block_length)
        },
        "embeddings": embeddings
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
    bounds = ox.utils_geo.bbox_from_point(
        (G.graph['center_lat'], G.graph['center_lon']),
        dist=G.graph['dist']
    )
    area_sqkm = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 111) ** 2
    
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
        "connectivity_index": float(connectivity_index),
        "network_density": float(network_density),
        "organic_ratio": float(organic_ratio),
        "intersection_density": float(node_count / area_sqkm if area_sqkm > 0 else 0)
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