"""Core metrics calculation module."""

import numpy as np
import networkx as nx
import geopandas as gpd
from typing import Dict, Any, List, Tuple
from shapely.geometry import Point, LineString, Polygon
import osmnx as ox
import scipy.stats as stats
from scipy import stats
from ..utils.area import calculate_area_sqm
from ..utils.formatting import (
    round_float,
    LENGTH_DECIMALS,
    DENSITY_DECIMALS,
    RATIO_DECIMALS,
    ANGLE_DECIMALS,
    PERCENT_DECIMALS,
    AREA_DECIMALS
)
from ..exceptions import GeoFeatureKitError

def calculate_network_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate network metrics using absolute measurements."""
    
    if not G:
        raise GeoFeatureKitError("Cannot calculate metrics for empty graph")
    
    # Basic metrics
    basic_metrics = {
        "total_street_length_meters": round_float(
            sum(d.get("length", 0) for _, _, d in G.edges(data=True)), LENGTH_DECIMALS
        ),
        "total_intersections": len([n for n, d in G.degree() if d > 2]),
        "total_dead_ends": len([n for n, d in G.degree() if d == 1]),
        "total_nodes": G.number_of_nodes(),
        "total_street_segments": G.number_of_edges()
    }
    
    # Get area in square meters from graph
    try:
        area_sqm = float(G.graph.get('area_sqm', 0))
        if area_sqm <= 0:
            # Calculate area from graph radius if available
            if 'dist' in G.graph:
                area_sqm = calculate_area_sqm(float(G.graph['dist']))
            else:
                raise GeoFeatureKitError("Graph missing both area_sqm and dist attributes")
    except (ValueError, TypeError) as e:
        raise GeoFeatureKitError(f"Invalid area value in graph: {e}")
    
    # Density metrics (per square meter)
    density_metrics = {
        "intersections_per_sqm": round_float(
            basic_metrics["total_intersections"] / area_sqm, DENSITY_DECIMALS
        ),
        "street_length_per_sqm": round_float(
            basic_metrics["total_street_length_meters"] / area_sqm, DENSITY_DECIMALS
        ),
        "nodes_per_sqm": round_float(
            basic_metrics["total_nodes"] / area_sqm, DENSITY_DECIMALS
        ),
        "units": "per_square_meter"
    }
    
    # Connectivity metrics
    G_undirected = G.to_undirected()
    node_degrees = [d for _, d in G_undirected.degree()]
    
    connectivity_metrics = {
        "average_connections_per_node": round_float(
            float(np.mean(node_degrees)) if node_degrees else None, RATIO_DECIMALS
        ),
        "number_of_disconnected_networks": len(list(nx.connected_components(G_undirected))),
        "number_of_street_loops": G.number_of_edges() - G.number_of_nodes() + len(list(nx.connected_components(G_undirected))),
        "streets_to_nodes_ratio": round_float(
            G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else None, RATIO_DECIMALS
        )
    }
    
    # Street pattern metrics
    bearings = [float(d.get("bearing", 0)) for _, _, d in G.edges(data=True) if "bearing" in d]
    bearings_array = np.array(bearings) if bearings else None
    
    # Count 90-degree intersections (within 10 degrees tolerance)
    ninety_degree_count = sum(1 for b in bearings if abs(b % 90) <= 10 or abs(b % 90) >= 80)
    
    # Get street segment lengths
    segment_lengths = [float(d.get("length", 0)) for _, _, d in G.edges(data=True) if "length" in d]
    segment_lengths_array = np.array(segment_lengths) if segment_lengths else None
    
    street_pattern_metrics = {
        "ninety_degree_intersection_count": ninety_degree_count,
        "most_common_street_bearing_degrees": round_float(
            float(stats.mode(bearings_array, keepdims=True)[0][0]) if bearings_array is not None and len(bearings) > 0 else None, 
            ANGLE_DECIMALS
        ),
        "street_bearing_standard_deviation_degrees": round_float(
            float(np.std(bearings_array)) if bearings_array is not None and len(bearings) > 0 else None, 
            ANGLE_DECIMALS
        ),
        "street_segment_length_distribution": {
            "minimum_meters": round_float(float(np.min(segment_lengths_array)) if segment_lengths_array is not None else None, LENGTH_DECIMALS),
            "maximum_meters": round_float(float(np.max(segment_lengths_array)) if segment_lengths_array is not None else None, LENGTH_DECIMALS),
            "median_meters": round_float(float(np.median(segment_lengths_array)) if segment_lengths_array is not None else None, LENGTH_DECIMALS),
            "standard_deviation_meters": round_float(float(np.std(segment_lengths_array)) if segment_lengths_array is not None else None, LENGTH_DECIMALS)
        }
    }
    
    return {
        "basic_metrics": basic_metrics,
        "density_metrics": density_metrics,
        "connectivity_metrics": connectivity_metrics,
        "street_pattern_metrics": street_pattern_metrics
    }

def calculate_poi_metrics(
    pois: gpd.GeoDataFrame,
    area_sqm: float
) -> Dict[str, Any]:
    """Calculate POI metrics using absolute measurements.

    Args:
        pois: GeoDataFrame containing POIs
        area_sqm: Area in square meters

    Returns:
        Dictionary containing POI metrics
    """
    # Validate inputs
    if not isinstance(pois, gpd.GeoDataFrame):
        raise GeoFeatureKitError("POIs must be a GeoDataFrame")
    
    try:
        area_sqm = float(area_sqm)
        if area_sqm <= 0:
            raise GeoFeatureKitError("Area must be positive")
    except (ValueError, TypeError) as e:
        raise GeoFeatureKitError(f"Invalid area value: {e}")

    if pois.empty:
        return {
            "absolute_counts": {
                "total_points_of_interest": 0,
                "counts_by_category": {}
            },
            "density_metrics": {
                "points_of_interest_per_sqm": None,
                "density_by_category": {},
                "units": "per_square_meter"
            },
            "distribution_metrics": {
                "unique_category_count": 0,
                "category_count_distribution": {
                    "most_frequent_count": None,
                    "least_frequent_count": None,
                    "median_category_count": None,
                    "category_count_standard_deviation": None
                },
                "largest_category_count": None,
                "largest_category_name": None,
                "largest_category_count_percent": None
            }
        }
    
    # Ensure amenity column exists
    if 'amenity' not in pois.columns:
        raise GeoFeatureKitError("POIs must have an 'amenity' column")
    
    # Calculate category counts, handling missing values
    category_counts = pois['amenity'].fillna('unknown').value_counts().to_dict()
    
    # Absolute counts
    absolute_counts = {
        "total_points_of_interest": len(pois),
        "counts_by_category": {
            f"total_{cat}_places": count
            for cat, count in category_counts.items()
        }
    }
    
    # Density metrics (per square meter)
    density_metrics = {
        "points_of_interest_per_sqm": round_float(
            len(pois) / area_sqm, DENSITY_DECIMALS
        ),
        "density_by_category": {
            f"{cat}_places_per_sqm": round_float(
                count / area_sqm, DENSITY_DECIMALS
            ) if round_float(count / area_sqm, DENSITY_DECIMALS) > 0 else round_float(count / area_sqm, 6)
            for cat, count in category_counts.items()
        },
        "units": "per_square_meter"
    }
    
    # Distribution metrics
    counts = np.array(list(category_counts.values()))
    distribution_metrics = {
        "unique_category_count": len(category_counts),
        "category_count_distribution": {
            "most_frequent_count": int(np.max(counts)) if len(counts) > 0 else None,
            "least_frequent_count": int(np.min(counts)) if len(counts) > 0 else None,
            "median_category_count": round_float(float(np.median(counts)) if len(counts) > 0 else None, RATIO_DECIMALS),
            "category_count_standard_deviation": round_float(float(np.std(counts)) if len(counts) > 0 else None, RATIO_DECIMALS)
        }
    }
    
    if counts.size > 0:
        max_category = max(category_counts.items(), key=lambda x: x[1])
        distribution_metrics.update({
            "largest_category_count": int(max_category[1]),
            "largest_category_name": str(max_category[0]),
            "largest_category_count_percent": round_float(
                max_category[1] * 100 / len(pois), PERCENT_DECIMALS
            )
        })
    else:
        distribution_metrics.update({
            "largest_category_count": None,
            "largest_category_name": None,
            "largest_category_count_percent": None
        })
    
    return {
        "absolute_counts": absolute_counts,
        "density_metrics": density_metrics,
        "distribution_metrics": distribution_metrics
    }

def calculate_pedestrian_network_metrics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate pedestrian network metrics using absolute measurements."""
    
    # Calculate intersection spacings
    intersection_spacings = []
    dead_end_length = 0
    dead_end_count = 0
    
    for node, degree in G.degree():
        if degree > 2:  # Intersection
            # Get distances to neighboring intersections
            for neighbor in G.neighbors(node):
                if G.degree(neighbor) > 2:  # Also an intersection
                    if 'length' in G[node][neighbor][0]:
                        intersection_spacings.append(
                            G[node][neighbor][0]['length']
                        )
        elif degree == 1:  # Dead end
            dead_end_count += 1
            # Sum length of dead-end street segments
            for neighbor in G.neighbors(node):
                if 'length' in G[node][neighbor][0]:
                    dead_end_length += G[node][neighbor][0]['length']
    
    spacing_metrics = {
        "minimum": round_float(min(intersection_spacings) if intersection_spacings else 0, LENGTH_DECIMALS),
        "maximum": round_float(max(intersection_spacings) if intersection_spacings else 0, LENGTH_DECIMALS),
        "median": round_float(float(np.median(intersection_spacings)) if intersection_spacings else 0, LENGTH_DECIMALS),
        "standard_deviation": round_float(float(np.std(intersection_spacings)) if intersection_spacings else 0, LENGTH_DECIMALS)
    }
    
    return {
        "intersection_spacing_meters": spacing_metrics,
        "dead_end_street_count": dead_end_count,
        "total_dead_end_street_length_meters": round_float(dead_end_length, LENGTH_DECIMALS)
    }

def calculate_land_use_metrics(
    land_use: gpd.GeoDataFrame,
    area_sqm: float
) -> Dict[str, Any]:
    """Calculate land use metrics using absolute measurements.

    Args:
        land_use: GeoDataFrame containing land use polygons
        area_sqm: Area in square meters

    Returns:
        Dictionary containing land use metrics
    """
    if land_use.empty:
        return {
            "area_measurements": {
                "total_area_sqm": round_float(area_sqm, AREA_DECIMALS),
                "residential_area_sqm": 0,
                "commercial_area_sqm": 0,
                "mixed_use_area_sqm": 0,
                "open_space_area_sqm": 0
            },
            "land_use_boundaries": {
                "total_boundary_length_meters": 0,
                "residential_boundary_length_meters": 0,
                "commercial_boundary_length_meters": 0,
                "mixed_use_boundary_length_meters": 0,
                "open_space_boundary_length_meters": 0
            },
            "land_use_percentages": {
                "residential_percent": 0,
                "commercial_percent": 0,
                "mixed_use_percent": 0,
                "open_space_percent": 0
            }
        }
    
    # Project to a local UTM zone for accurate area calculations
    # Get centroid of all geometries
    centroid = land_use.geometry.union_all().centroid
    # Get UTM zone number for the centroid
    utm_zone = int(((centroid.x + 180) / 6) + 1)
    # Get hemisphere (north or south)
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    # Construct EPSG code for appropriate UTM zone
    epsg = f"326{utm_zone:02d}" if hemisphere == 'north' else f"327{utm_zone:02d}"
    # Project to UTM
    land_use_proj = land_use.to_crs(epsg=epsg)
    
    # Calculate areas by type
    area_by_type = {}
    boundary_lengths = {}
    percentages = {}
    
    for land_type in ['residential', 'commercial', 'mixed_use', 'open_space']:
        mask = land_use_proj['landuse'] == land_type
        if mask.any():
            area = land_use_proj[mask].geometry.area.sum() / 10000  # Convert m² to hectares
            boundary = land_use_proj[mask].geometry.length.sum()
            percentage = (area / area_sqm * 100) if area_sqm > 0 else 0
        else:
            area = 0
            boundary = 0
            percentage = 0
            
        area_by_type[f"{land_type}_area_sqm"] = round_float(area, AREA_DECIMALS)
        boundary_lengths[f"{land_type}_boundary_length_meters"] = round_float(boundary, LENGTH_DECIMALS)
        percentages[f"{land_type}_percent"] = round_float(percentage, PERCENT_DECIMALS)
    
    total_area = land_use_proj.geometry.area.sum() / 10000  # Convert m² to hectares
    total_boundary = land_use_proj.geometry.length.sum()
    
    return {
        "area_measurements": {
            "total_area_sqm": round_float(total_area, AREA_DECIMALS),
            **area_by_type
        },
        "land_use_boundaries": {
            "total_boundary_length_meters": round_float(total_boundary, LENGTH_DECIMALS),
            **boundary_lengths
        },
        "land_use_percentages": percentages
    }

def calculate_data_quality_metrics(
    G: nx.MultiDiGraph,
    pois: gpd.GeoDataFrame,
    land_use: gpd.GeoDataFrame
) -> Dict[str, Any]:
    """Calculate data quality metrics."""
    
    # Simple completeness checks (presence of required attributes)
    network_complete = all(
        'length' in d and 'bearing' in d
        for _, _, d in G.edges(data=True)
    )
    poi_complete = not pois.empty and 'amenity' in pois.columns
    land_use_complete = not land_use.empty and 'landuse' in land_use.columns
    
    # Reliability scores (simplified example)
    network_reliable = 0.9  # Based on OSM data quality standards
    poi_reliable = 0.8     # POI data tends to be less reliable
    overall_reliable = 1.0  # Perfect score for now
    
    return {
        "data_completeness_percentages": {
            "percent_network_data_complete": round_float(network_complete * 100, PERCENT_DECIMALS),
            "percent_poi_data_complete": round_float(poi_complete * 100, PERCENT_DECIMALS),
            "percent_land_use_data_complete": round_float(land_use_complete * 100, PERCENT_DECIMALS)
        },
        "data_reliability_percentages": {
            "percent_network_data_reliable": round_float(network_reliable * 100, PERCENT_DECIMALS),
            "percent_poi_data_reliable": round_float(poi_reliable * 100, PERCENT_DECIMALS),
            "percent_overall_data_reliable": round_float(overall_reliable * 100, PERCENT_DECIMALS)
        }
    }

def calculate_all_metrics(
    G: nx.MultiDiGraph,
    pois: gpd.GeoDataFrame,
    land_use: gpd.GeoDataFrame
) -> Dict[str, Any]:
    """Calculate all metrics for a location."""
    
    # Calculate area in square meters
    radius_meters = G.graph.get('dist', 500)  # Default to 500m if not set
    area_sqm = calculate_area_sqm(radius_meters)
    
    # Create a copy of G with area information
    G = G.copy()
    G.graph['area_sqm'] = area_sqm
    
    # Get location information with defaults
    center_lat = G.graph.get('center_lat', 0)
    center_lon = G.graph.get('center_lon', 0)
    network_type = G.graph.get('network_type', 'all')
    
    return {
        "metadata": {
            "location": {
                "latitude": center_lat,
                "longitude": center_lon
            },
            "radius_meters": radius_meters,
            "network_type": network_type,
            "area_sqm": round_float(area_sqm, AREA_DECIMALS)
        },
        "metrics": {
            "network_metrics": calculate_network_metrics(G),
            "poi_metrics": calculate_poi_metrics(pois, area_sqm),
            "pedestrian_network": calculate_pedestrian_network_metrics(G),
            "land_use_metrics": calculate_land_use_metrics(land_use, area_sqm),
            "data_quality_metrics": calculate_data_quality_metrics(G, pois, land_use)
        }
    } 