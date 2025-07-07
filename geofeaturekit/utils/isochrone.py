"""Isochrone calculation utilities for accessibility analysis."""

import numpy as np
import networkx as nx
import geopandas as gpd
from typing import Dict, Any, Optional, List, Tuple
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import osmnx as ox
from ..exceptions.errors import GeoFeatureKitError
from ..utils.formatting import round_float, LENGTH_DECIMALS, AREA_DECIMALS, RATIO_DECIMALS
from ..utils.area import calculate_area_hectares
from ..utils.network import download_network
from ..utils.poi import download_pois
from ..utils.progress import EnhancedProgressTracker
from ..core.metrics import calculate_all_metrics


def calculate_isochrone_distance(travel_time_minutes: float, speed_kmh: float) -> float:
    """Calculate maximum travel distance for given time and speed.
    
    Args:
        travel_time_minutes: Maximum travel time in minutes
        speed_kmh: Travel speed in km/h
        
    Returns:
        Maximum travel distance in meters
    """
    # Convert km/h to m/min
    meters_per_minute = (speed_kmh * 1000) / 60
    max_distance_meters = travel_time_minutes * meters_per_minute
    return max_distance_meters


def create_isochrone_polygon(
    latitude: float,
    longitude: float,
    travel_time_minutes: float,
    mode: str,
    speed_kmh: float,
    network_type: Optional[str] = None
) -> Optional[Polygon]:
    """Create an isochrone polygon for accessibility analysis.
    
    Args:
        latitude: Center point latitude
        longitude: Center point longitude
        travel_time_minutes: Maximum travel time in minutes
        mode: Transportation mode ('walk', 'bike', 'drive')
        speed_kmh: Travel speed in km/h
        network_type: OSM network type override
        
    Returns:
        Shapely Polygon representing the isochrone area, or None if calculation fails
    """
    try:
        # Map mode to network type if not specified
        if network_type is None:
            network_type_mapping = {
                'walk': 'walk',
                'bike': 'bike',
                'drive': 'drive'
            }
            network_type = network_type_mapping.get(mode, 'all')
        
        # Calculate maximum distance
        max_distance = calculate_isochrone_distance(travel_time_minutes, speed_kmh)
        
        # For network-based routing, we need to get the actual network
        # and calculate travel times along the network
        try:
            # Get street network for the area
            G = ox.graph_from_point(
                (latitude, longitude),
                dist=max_distance * 1.2,  # Get larger network for edge effects
                network_type=network_type
            )
            
            # Add travel speeds to edges
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            
            # Override speeds based on mode
            for u, v, k, data in G.edges(data=True, keys=True):
                # Set speed based on mode
                if mode == 'walk':
                    data['speed_kph'] = speed_kmh
                elif mode == 'bike':
                    data['speed_kph'] = min(speed_kmh, data.get('speed_kph', speed_kmh))
                elif mode == 'drive':
                    data['speed_kph'] = data.get('speed_kph', speed_kmh)
                
                # Recalculate travel time
                data['travel_time'] = data['length'] / (data['speed_kph'] * 1000 / 3600)
            
            # Find center node
            center_node = ox.nearest_nodes(G, longitude, latitude)
            
            # Calculate travel times from center to all other nodes
            travel_times = nx.single_source_dijkstra_path_length(
                G, center_node, weight='travel_time'
            )
            
            # Filter nodes within travel time limit
            reachable_nodes = [
                node for node, time in travel_times.items()
                if time <= travel_time_minutes * 60  # Convert minutes to seconds
            ]
            
            if not reachable_nodes:
                return None
            
            # Get coordinates of reachable nodes
            reachable_coords = [
                (G.nodes[node]['x'], G.nodes[node]['y'])
                for node in reachable_nodes
            ]
            
            # Create convex hull around reachable points
            from scipy.spatial import ConvexHull
            if len(reachable_coords) >= 3:
                hull = ConvexHull(reachable_coords)
                hull_coords = [reachable_coords[i] for i in hull.vertices]
                hull_coords.append(hull_coords[0])  # Close the polygon
                return Polygon(hull_coords)
            else:
                # Fallback to circular buffer
                center_point = Point(longitude, latitude)
                return center_point.buffer(max_distance / 111320)  # Approximate degrees
                
        except Exception as e:
            # Fallback to simple circular buffer if network routing fails
            print(f"  Warning: Network routing failed for {mode}, using circular approximation: {str(e)}")
            center_point = Point(longitude, latitude)
            # Convert meters to approximate degrees (rough approximation)
            buffer_degrees = max_distance / 111320
            return center_point.buffer(buffer_degrees)
            
    except Exception as e:
        print(f"  Warning: Isochrone calculation failed: {str(e)}")
        return None


def extract_isochrone_features(
    latitude: float,
    longitude: float,
    travel_time_minutes: float,
    mode: str,
    speed_kmh: float,
    verbose: bool = False,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Extract urban features within an isochrone boundary.
    
    Args:
        latitude: Center point latitude
        longitude: Center point longitude
        travel_time_minutes: Maximum travel time in minutes
        mode: Transportation mode ('walk', 'bike', 'drive')
        speed_kmh: Travel speed in km/h
        verbose: Enable verbose console output (default: False)
        progress_callback: Optional callback function(message: str, progress: float)
        
    Returns:
        Dictionary containing isochrone-based urban features and metrics
    """
    # Create isochrone polygon
    isochrone_polygon = create_isochrone_polygon(
        latitude, longitude, travel_time_minutes, mode, speed_kmh
    )
    
    if isochrone_polygon is None:
        return {
            "isochrone_info": {
                "mode": mode,
                "travel_time_minutes": travel_time_minutes,
                "speed_kmh": speed_kmh,
                "area_sqm": 0,
                "calculation_method": "failed"
            },
            "network_metrics": {},
            "poi_metrics": {},
            "error": "Failed to calculate isochrone"
        }
    
    # Calculate isochrone area
    # Convert from degrees to square meters (rough approximation)
    area_deg_sq = isochrone_polygon.area
    # Use approximate conversion: 1 degree ≈ 111320 meters at equator
    area_sqm = area_deg_sq * (111320 ** 2) * np.cos(np.radians(latitude))
    
    location_desc = f"({latitude:.4f}, {longitude:.4f}) - {mode.title()} {travel_time_minutes}min"
    
    try:
        if progress_callback:
            progress_callback(f"Starting {mode} isochrone analysis", 0.0)
        elif verbose:
            print(f"Starting {mode} isochrone analysis for {location_desc}")
        
        # Phase 1: Data Download (70% of total time)
        if progress_callback:
            progress_callback("Downloading street network", 0.1)
        elif verbose:
            print("Downloading street network...")
        
        # Get network using point-based approach (more stable)
        try:
            # Estimate radius from isochrone area
            estimated_radius = np.sqrt(area_sqm / np.pi)
            
            # Get network from center point with estimated radius
            G = ox.graph_from_point(
                (latitude, longitude),
                dist=estimated_radius,
                network_type=mode if mode != 'walk' else 'walk'
            )
            
            # Filter network to only include nodes within isochrone (optional - for accuracy)
            if G is not None and G.number_of_nodes() > 0:
                nodes_to_remove = []
                for node in G.nodes():
                    node_point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
                    if not isochrone_polygon.contains(node_point):
                        nodes_to_remove.append(node)
                
                # Only remove nodes if we're not removing everything
                if len(nodes_to_remove) < G.number_of_nodes() * 0.9:
                    G.remove_nodes_from(nodes_to_remove)
            
            # Add area metadata
            if G is not None:
                G.graph['area_sqm'] = area_sqm
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Network download failed: {str(e)}")
            G = None
        
        if progress_callback:
            progress_callback("Downloading points of interest", 0.4)
        elif verbose:
            print("Downloading points of interest...")
        
        # Get POIs using point-based approach (more stable)
        try:
            # Estimate radius from isochrone area  
            estimated_radius = np.sqrt(area_sqm / np.pi)
            
            # Use point-based download instead of bounding box
            pois = ox.features_from_point(
                (latitude, longitude),
                dist=estimated_radius,
                tags={
                    'amenity': True,
                    'leisure': True,
                    'shop': True,
                    'tourism': True,
                    'historic': True,
                    'office': True,
                    'public_transport': True,
                    'healthcare': True,
                    'education': True,
                    'natural': True,
                    'waterway': True
                }
            )
            
            # Filter POIs to only include those within isochrone
            if not pois.empty:
                # Filter to just points first
                point_pois = pois[pois.geometry.type == 'Point'].copy()
                # Then filter by isochrone boundary
                pois_within = point_pois[point_pois.geometry.within(isochrone_polygon)].copy()
            else:
                pois_within = gpd.GeoDataFrame(columns=['amenity'], geometry=[])
                
        except Exception as e:
            if verbose:
                print(f"  Warning: POI download failed: {str(e)}")
            pois_within = gpd.GeoDataFrame(columns=['amenity'], geometry=[])
        
        # Phase 2: Calculate Metrics (25% of total time)
        if progress_callback:
            progress_callback("Computing accessibility metrics", 0.7)
        elif verbose:
            print("Computing accessibility metrics...")
        
        # Calculate metrics using existing functions
        metrics = calculate_all_metrics(G, pois_within, area_sqm)
        
        # Phase 3: Format Results (5% of total time)
        if progress_callback:
            progress_callback("Finalizing results", 0.9)
        elif verbose:
            print("Finalizing results...")
        
        # Add isochrone-specific information
        result = {
            "isochrone_info": {
                "mode": mode,
                "travel_time_minutes": travel_time_minutes,
                "speed_kmh": speed_kmh,
                "area_sqm": round_float(area_sqm, AREA_DECIMALS),
                "calculation_method": "network_based" if G is not None else "circular_approximation",
                "accessible_nodes": G.number_of_nodes() if G is not None else 0,
                "accessible_pois": len(pois_within)
            },
            **metrics
        }
        
        if progress_callback:
            progress_callback("Complete", 1.0)
        elif verbose:
            print("Isochrone analysis complete")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"Error: Failed to extract isochrone features: {str(e)}")
        raise GeoFeatureKitError(f"Failed to extract isochrone features: {str(e)}")


def validate_speed_config(speed_config: Dict[str, float]) -> None:
    """Validate speed configuration dictionary.
    
    Args:
        speed_config: Dictionary with speed settings for each mode
        
    Raises:
        ValueError: If speed configuration is invalid
        TypeError: If speed values are not numeric
    """
    required_modes = ['walk', 'bike', 'drive']
    
    for mode in required_modes:
        if mode not in speed_config:
            raise ValueError(f"Missing '{mode}' in speed_config.")
        
        if not isinstance(speed_config[mode], (int, float)):
            raise TypeError(f"Speed for '{mode}' must be numeric (km/h).")
        
        if speed_config[mode] <= 0:
            raise ValueError(f"Speed for '{mode}' must be positive.")


def get_default_speed_config() -> Dict[str, float]:
    """Get default speed configuration.
    
    Returns:
        Dictionary with default speeds for each transportation mode
    """
    return {
        'walk': 5.0,   # km/h - Average walking speed
        'bike': 15.0,  # km/h - Average cycling speed
        'drive': 40.0  # km/h - Average urban driving speed
    } 