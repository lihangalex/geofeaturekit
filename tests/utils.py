"""Test utilities for generating test data and validating results."""

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
from typing import Dict, Any, Tuple, List

def _add_edge_attributes(G: nx.MultiDiGraph, u: Tuple[int, int], v: Tuple[int, int]) -> None:
    """Add edge attributes to a directed edge.
    
    Args:
        G: NetworkX graph
        u: Source node
        v: Target node
    """
    # Calculate bearing
    dx = G.nodes[v]['x'] - G.nodes[u]['x']
    dy = G.nodes[v]['y'] - G.nodes[u]['y']
    bearing = np.degrees(np.arctan2(dx, dy)) % 360
    
    # Calculate length
    length = np.sqrt(dx**2 + dy**2)
    
    # Add edge attributes
    G[u][v][0].update({
        'length': float(length),
        'bearing': float(bearing),
        'osmid': hash(f"{u}-{v}"),
        'highway': 'residential'
    })

def create_test_graph(
    num_nodes: int = 10,
    grid_layout: bool = True,
    radius_meters: float = 500
) -> nx.MultiDiGraph:
    """Create a test graph with known properties.
    
    Args:
        num_nodes: Number of nodes in the graph
        grid_layout: If True, creates a grid layout, otherwise random
        radius_meters: Radius of the analysis area in meters
        
    Returns:
        NetworkX graph with known properties
    """
    if grid_layout:
        # Create a grid graph
        n = int(np.sqrt(num_nodes))
        G = nx.grid_2d_graph(n, n)
        # Convert to directed multigraph
        G_directed = nx.MultiDiGraph()
        
        # Add coordinates (scale to fit in radius)
        scale = radius_meters / (n - 1) if n > 1 else radius_meters
        pos = {(i, j): (i * scale - radius_meters/2, j * scale - radius_meters/2) 
               for i in range(n) for j in range(n)}
        
        # Add node attributes
        for node in G.nodes():
            G_directed.add_node(node)
            G_directed.nodes[node]['x'] = float(pos[node][0])
            G_directed.nodes[node]['y'] = float(pos[node][1])
            G_directed.nodes[node]['osmid'] = hash(str(node))
            G_directed.nodes[node]['street_count'] = 0
        
        # Add edges in both directions (no diagonals)
        for i in range(n):
            for j in range(n):
                # Add horizontal edges
                if i < n-1:
                    u = (i, j)
                    v = (i+1, j)
                    G_directed.add_edge(u, v)
                    G_directed.add_edge(v, u)
                    _add_edge_attributes(G_directed, u, v)
                    _add_edge_attributes(G_directed, v, u)
                    # Update street counts for both nodes
                    G_directed.nodes[u]['street_count'] += 1
                    G_directed.nodes[v]['street_count'] += 1
                # Add vertical edges
                if j < n-1:
                    u = (i, j)
                    v = (i, j+1)
                    G_directed.add_edge(u, v)
                    G_directed.add_edge(v, u)
                    _add_edge_attributes(G_directed, u, v)
                    _add_edge_attributes(G_directed, v, u)
                    # Update street counts for both nodes
                    G_directed.nodes[u]['street_count'] += 1
                    G_directed.nodes[v]['street_count'] += 1
        
        G = G_directed
    else:
        # Modified random graph generation to ensure better connectivity
        # Generate random points within the radius
        points = np.random.uniform(-radius_meters/2, radius_meters/2, (num_nodes, 2))
        
        # Create Delaunay triangulation
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        
        # Create graph from triangulation
        G = nx.MultiDiGraph()
        
        # Add nodes
        for i, (x, y) in enumerate(points):
            G.add_node(i, x=float(x), y=float(y), osmid=hash(str(i)), street_count=0)
        
        # Add edges from triangulation (both directions)
        edges_added = set()
        for simplex in tri.simplices:
            for i in range(3):
                u, v = simplex[i], simplex[(i+1)%3]
                if (u, v) not in edges_added and (v, u) not in edges_added:
                    # Add both directions
                    G.add_edge(u, v)
                    G.add_edge(v, u)
                    _add_edge_attributes(G, u, v)
                    _add_edge_attributes(G, v, u)
                    edges_added.add((u, v))
                    edges_added.add((v, u))
                    # Update street count
                    G.nodes[u]['street_count'] += 1
                    G.nodes[v]['street_count'] += 1
        
        # Add more edges to ensure good connectivity
        # We want at least 2.0 * num_nodes undirected edges
        target_undirected_edges = int(2.0 * num_nodes)
        current_undirected_edges = len(edges_added) // 2
        
        if current_undirected_edges < target_undirected_edges:
            # Add edges to nearest neighbors until we reach target
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            
            # Query k nearest neighbors for each point
            k = min(6, num_nodes - 1)  # Increased from 4 to 6 for better connectivity
            distances, indices = tree.query(points, k=k+1)  # k+1 because first point is self
            
            # Add edges to nearest neighbors
            for i in range(num_nodes):
                for j in range(1, k+1):  # Skip first neighbor (self)
                    neighbor = indices[i, j]
                    if i != neighbor and (i, neighbor) not in edges_added:
                        G.add_edge(i, neighbor)
                        G.add_edge(neighbor, i)
                        _add_edge_attributes(G, i, neighbor)
                        _add_edge_attributes(G, neighbor, i)
                        edges_added.add((i, neighbor))
                        edges_added.add((neighbor, i))
                        G.nodes[i]['street_count'] += 1
                        G.nodes[neighbor]['street_count'] += 1
                        
                        current_undirected_edges += 1
                        if current_undirected_edges >= target_undirected_edges:
                            break
                if current_undirected_edges >= target_undirected_edges:
                    break
            
            # If we still don't have enough edges, add random edges
            if current_undirected_edges < target_undirected_edges:
                nodes = list(G.nodes())
                attempts = 0
                max_attempts = num_nodes * 10  # Prevent infinite loops
                
                while current_undirected_edges < target_undirected_edges and attempts < max_attempts:
                    u = np.random.choice(nodes)
                    v = np.random.choice(nodes)
                    
                    if u != v and (u, v) not in edges_added:
                        G.add_edge(u, v)
                        G.add_edge(v, u)
                        _add_edge_attributes(G, u, v)
                        _add_edge_attributes(G, v, u)
                        edges_added.add((u, v))
                        edges_added.add((v, u))
                        G.nodes[u]['street_count'] += 1
                        G.nodes[v]['street_count'] += 1
                        current_undirected_edges += 1
                    
                    attempts += 1
    
    # Add graph attributes
    G.graph['crs'] = 'epsg:4326'
    G.graph['dist'] = radius_meters
    G.graph['area_sqm'] = np.pi * radius_meters**2
    
    return G

def create_test_pois(
    num_pois: int = 100,
    radius_meters: float = 500,
    categories: List[str] = None,
    pattern: str = 'random'
) -> gpd.GeoDataFrame:
    """Create test POIs with known properties.
    
    Args:
        num_pois: Number of POIs to generate
        radius_meters: Radius of the analysis area in meters
        categories: List of POI categories (if None, uses default categories)
        pattern: Spatial pattern ('random', 'clustered', or 'grid')
        
    Returns:
        GeoDataFrame containing POIs
    """
    if categories is None:
        categories = ['restaurant', 'cafe', 'bar', 'shop', 'school']
    
    if pattern == 'random':
        # Generate random points within circle
        theta = np.random.uniform(0, 2*np.pi, num_pois)
        r = np.sqrt(np.random.uniform(0, 1, num_pois)) * radius_meters
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    elif pattern == 'clustered':
        # Generate clustered points
        num_clusters = min(5, max(1, num_pois // 5))  # Ensure at least 1 cluster
        points_per_cluster = num_pois // num_clusters
        remainder = num_pois % num_clusters
        
        # Generate cluster centers within the circle
        cluster_theta = np.random.uniform(0, 2*np.pi, num_clusters)
        cluster_r = np.sqrt(np.random.uniform(0, 0.5, num_clusters)) * radius_meters  # Keep clusters within inner 70% of radius
        cluster_centers_x = cluster_r * np.cos(cluster_theta)
        cluster_centers_y = cluster_r * np.sin(cluster_theta)
        
        x = []
        y = []
        for i, (cx, cy) in enumerate(zip(cluster_centers_x, cluster_centers_y)):
            # Add one extra point to this cluster if we have remainder points
            cluster_size = points_per_cluster + (1 if i < remainder else 0)
            if cluster_size > 0:  # Only generate points if cluster_size > 0
                # Generate points with tighter normal distribution around cluster center
                cluster_points = np.random.normal(loc=[cx, cy], scale=radius_meters/20, size=(cluster_size, 2))  # Reduced scale from /10 to /20
                # Clip points to stay within radius
                distances = np.sqrt(np.sum(cluster_points**2, axis=1))
                outside_radius = distances > radius_meters
                if outside_radius.any():
                    # Scale points back to radius
                    cluster_points[outside_radius] *= (radius_meters / distances[outside_radius, np.newaxis])
                x.extend(cluster_points[:, 0])
                y.extend(cluster_points[:, 1])
    else:  # grid
        # Calculate grid size to get closest to num_pois
        n = int(np.sqrt(num_pois))
        spacing = radius_meters * 2 / (n + 1)  # Ensure points are within radius
        x = []
        y = []
        for i in range(n):
            for j in range(n):
                # Add small random offset to avoid perfect grid
                offset_x = np.random.uniform(-spacing/10, spacing/10)
                offset_y = np.random.uniform(-spacing/10, spacing/10)
                x.append(-radius_meters + (i+1) * spacing + offset_x)
                y.append(-radius_meters + (j+1) * spacing + offset_y)
        
        # Trim to exact number of points
        x = x[:num_pois]
        y = y[:num_pois]
    
    # Create GeoDataFrame
    geometry = [Point(x, y) for x, y in zip(x, y)]
    names = [f"POI_{i}" for i in range(len(x))]
    amenities = np.random.choice(categories, size=len(x))
    
    data = pd.DataFrame({
        'name': names,
        'amenity': amenities,
        'geometry': geometry
    })
    
    gdf = gpd.GeoDataFrame(data, crs='epsg:4326')
    return gdf

def calculate_expected_metrics(
    G: nx.MultiDiGraph,
    pois: gpd.GeoDataFrame
) -> Dict[str, Any]:
    """Calculate expected metrics for test data validation.
    
    Args:
        G: NetworkX graph
        pois: GeoDataFrame containing POIs
        
    Returns:
        Dictionary containing expected metrics
    """
    # Calculate basic network metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    total_length = sum(d['length'] for _, _, d in G.edges(data=True))
    
    # Calculate intersection counts
    # For directed graphs, street_count represents the number of unique streets
    # An intersection has 3 or more streets meeting
    # A dead end has exactly 1 street
    intersections = len([n for n, d in G.nodes(data=True) if d.get('street_count', 0) >= 3])
    dead_ends = len([n for n, d in G.nodes(data=True) if d.get('street_count', 0) == 1])
    
    # Calculate area
    radius_meters = float(G.graph['dist'])
    area_sqm = np.pi * radius_meters**2
    area_sqkm = area_sqm / 1_000_000
    
    # Calculate POI metrics
    total_pois = len(pois)
    category_counts = pois['amenity'].value_counts().to_dict()
    
    return {
        "network_metrics": {
            "basic_metrics": {
                "total_nodes": num_nodes,
                "total_street_segments": num_edges,
                "total_intersections": intersections,
                "total_dead_ends": dead_ends,
                "total_street_length_meters": total_length
            },
            "density_metrics": {
                "intersections_per_sqkm": intersections / area_sqkm,
                "street_length_per_sqkm": total_length / 1000 / area_sqkm
            }
        },
        "poi_metrics": {
            "absolute_counts": {
                "total_points_of_interest": total_pois,
                "counts_by_category": {
                    f"total_{cat}_places": {
                        "count": count,
                        "percentage": count/total_pois * 100
                    }
                    for cat, count in category_counts.items()
                }
            },
            "density_metrics": {
                "points_of_interest_per_sqkm": total_pois / area_sqkm
            }
        }
    }

def assert_metrics_match(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    rtol: float = 1e-5
) -> None:
    """Assert that actual metrics match expected values within tolerance.
    
    Args:
        actual: Dictionary of actual metrics
        expected: Dictionary of expected metrics
        rtol: Relative tolerance for floating point comparisons
    """
    def _compare_dicts(d1: Dict[str, Any], d2: Dict[str, Any], path: str = "") -> None:
        for key in d2:
            if key not in d1:
                raise AssertionError(f"Missing key {path}.{key} in actual metrics")
            
            if isinstance(d2[key], dict):
                if not isinstance(d1[key], dict):
                    raise AssertionError(f"Value at {path}.{key} should be a dictionary")
                _compare_dicts(d1[key], d2[key], f"{path}.{key}")
            elif isinstance(d2[key], (int, float)):
                if not isinstance(d1[key], (int, float)):
                    raise AssertionError(f"Value at {path}.{key} should be numeric")
                if not np.isclose(d1[key], d2[key], rtol=rtol):
                    raise AssertionError(
                        f"Value mismatch at {path}.{key}: "
                        f"expected {d2[key]}, got {d1[key]}"
                    )
    
    _compare_dicts(actual, expected) 