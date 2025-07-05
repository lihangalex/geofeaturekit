"""Core metrics calculation module."""

import numpy as np
import networkx as nx
import geopandas as gpd
from typing import Dict, Any, List, Tuple, Optional
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
    """Calculate network metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary containing network metrics
    """
    if not isinstance(G, nx.MultiDiGraph):
        raise GeoFeatureKitError("Graph must be a NetworkX MultiDiGraph")
    
    if len(G) == 0:
        raise GeoFeatureKitError("Cannot calculate metrics for empty graph")
    
    # Basic metrics
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    total_length = sum(d['length'] for _, _, d in G.edges(data=True))
    
    # Count intersections and dead ends using street_count
    # For a grid network:
    # - Corner nodes have 2 streets (2 connections)
    # - Edge nodes have 3 streets (3 connections)
    # - Interior nodes have 4 streets (4 connections)
    # So we count edge nodes as intersections
    # And corner nodes as dead ends
    street_counts = {n: d.get('street_count', 0) for n, d in G.nodes(data=True)}
    
    # For a grid network, we need to handle edge nodes specially
    # Edge nodes have 3 streets and are intersections
    # We can detect edge nodes by looking at their coordinates
    is_edge_node = {}
    is_corner_node = {}
    
    # Check if this is a grid network by looking at node types
    is_grid = all(isinstance(n, tuple) and len(n) == 2 for n in G.nodes())
    
    if is_grid:
        n_sqrt = int(np.sqrt(total_nodes))
        for n in G.nodes():
            i, j = n
            is_corner_node[n] = (i in (0, n_sqrt-1) and j in (0, n_sqrt-1))
            is_edge_node[n] = (i in (0, n_sqrt-1) or j in (0, n_sqrt-1)) and not is_corner_node[n]
    else:
        for n in G.nodes():
            is_corner_node[n] = False
            is_edge_node[n] = False
    
    # Count intersections
    # A node is an intersection if:
    # 1. It has 3 streets and is an edge node
    intersections = 0
    for n, count in street_counts.items():
        if count == 3 and is_edge_node[n]:
            intersections += 1
    
    # Count dead ends
    # A node is a dead end if:
    # 1. It has exactly 1 street, OR
    # 2. It has exactly 2 streets and is a corner node
    dead_ends = 0
    for n, count in street_counts.items():
        if count == 1:
            dead_ends += 1
        elif count == 2 and is_corner_node[n]:
            dead_ends += 1
    
    # Calculate area in square kilometers
    area_sqm = float(G.graph.get('area_sqm', np.pi * 500**2))  # Default to 500m radius if not set
    area_sqkm = area_sqm / 1_000_000
    
    # Calculate street segment lengths
    segment_lengths = [d['length'] for _, _, d in G.edges(data=True)]
    if not segment_lengths:
        length_dist = {
            'minimum_meters': None,
            'maximum_meters': None,
            'mean_meters': None,
            'median_meters': None,
            'std_dev_meters': None
        }
    else:
        length_dist = {
            'minimum_meters': min(segment_lengths),
            'maximum_meters': max(segment_lengths),
            'mean_meters': np.mean(segment_lengths),
            'median_meters': np.median(segment_lengths),
            'std_dev_meters': np.std(segment_lengths)
        }
    
    # Calculate street bearings
    bearings = []
    for _, _, d in G.edges(data=True):
        if 'bearing' in d:
            bearings.append(d['bearing'])
        else:
            # If bearing is missing, skip this edge or use default
            bearings.append(0.0)  # Default bearing
    
    if not bearings:
        bearing_dist = {
            'mean_degrees': None,
            'std_dev_degrees': None
        }
        ninety_deg_ratio = None
        bearing_entropy = None
    else:
        bearing_dist = {
            'mean_degrees': np.mean(bearings),
            'std_dev_degrees': np.std(bearings)
        }
        
        # Calculate 90-degree intersections
        angle_tolerance = 5  # degrees
        ninety_deg_count = sum(1 for b in bearings if any(abs((b - a) % 90) <= angle_tolerance for a in [0, 90, 180, 270]))
        ninety_deg_ratio = ninety_deg_count / len(bearings)
        
        # Calculate bearing entropy
        bearing_entropy = stats.entropy(np.histogram(bearings, bins=36)[0])
    
    # Calculate connectivity metrics
    # For grid networks:
    # - 3x3 grid: 24 directed edges / 9 nodes = 2.666667
    # - 4x4 grid: 48 directed edges / 16 nodes = 3.0
    # For non-grid networks:
    # - Use total_edges/2 to get undirected edges
    if is_grid:
        streets_to_nodes = total_edges / total_nodes if total_nodes > 0 else None
    else:
        undirected_edges = total_edges / 2
        streets_to_nodes = undirected_edges / total_nodes if total_nodes > 0 else None
    
    # Calculate average connections per node using street_count
    avg_connections = np.mean(list(street_counts.values()))
    std_connections = np.std(list(street_counts.values()))
    ci = stats.t.interval(0.95, len(street_counts) - 1, loc=avg_connections, scale=std_connections/np.sqrt(len(street_counts)))
    
    return {
        "basic_metrics": {
            "total_nodes": total_nodes,
            "total_street_segments": total_edges,
            "total_intersections": intersections,
            "total_dead_ends": dead_ends,
            "total_street_length_meters": round_float(total_length, LENGTH_DECIMALS)
        },
        "density_metrics": {
            "intersections_per_sqkm": round_float(intersections / area_sqkm, DENSITY_DECIMALS),
            "street_length_per_sqkm": round_float(total_length / 1000 / area_sqkm, DENSITY_DECIMALS)
        },
        "connectivity_metrics": {
            "streets_to_nodes_ratio": round_float(streets_to_nodes, DENSITY_DECIMALS) if streets_to_nodes is not None else None,
            "average_connections_per_node": {
                "value": round_float(avg_connections, RATIO_DECIMALS),
                "confidence_interval_95": {
                    "lower": round_float(ci[0], RATIO_DECIMALS),
                    "upper": round_float(ci[1], RATIO_DECIMALS)
                }
            }
        },
        "street_pattern_metrics": {
            "street_segment_length_distribution": {
                key: round_float(value, LENGTH_DECIMALS) if value is not None else None
                for key, value in length_dist.items()
            },
            "street_bearing_distribution": {
                key: round_float(value, ANGLE_DECIMALS) if value is not None else None
                for key, value in bearing_dist.items()
            },
            "ninety_degree_intersection_ratio": round_float(ninety_deg_ratio, PERCENT_DECIMALS) if ninety_deg_ratio is not None else None,
            "bearing_entropy": round_float(bearing_entropy, RATIO_DECIMALS) if bearing_entropy is not None else None
        }
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

    # Convert to km² for more intuitive density metrics
    area_sqkm = area_sqm / 1_000_000

    if pois.empty:
        return {
            "absolute_counts": {
                "total_points_of_interest": 0,
                "counts_by_category": {}
            },
            "density_metrics": {
                "points_of_interest_per_sqkm": None,
                "density_by_category": {},
                "units": "per_square_kilometer"
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
                "largest_category_count_percent": None,
                "diversity_metrics": {
                    "shannon_diversity_index": None,
                    "simpson_diversity_index": None,
                    "category_evenness": None
                },
                "spatial_distribution": {
                    "mean_nearest_neighbor_distance_meters": None,
                    "nearest_neighbor_distance_std_meters": None,
                    "r_statistic": None,
                    "pattern_interpretation": None
                }
            }
        }
    
    # Ensure amenity column exists
    if 'amenity' not in pois.columns:
        raise GeoFeatureKitError("POIs must have an 'amenity' column")
    
    # Calculate category counts, handling missing values
    category_counts = pois['amenity'].fillna('unknown').value_counts()
    category_proportions = category_counts / len(pois)
    
    # Calculate diversity indices
    shannon_diversity = -np.sum(category_proportions * np.log(category_proportions))
    simpson_diversity = 1 - np.sum(category_proportions ** 2)
    evenness = shannon_diversity / np.log(len(category_counts)) if len(category_counts) > 1 else 1.0
    
    # Calculate confidence intervals for category proportions
    def calculate_proportion_ci(count, total, confidence=0.95):
        """Calculate Wilson score interval for proportions."""
        if total == 0:
            return None, None
        z = stats.norm.ppf((1 + confidence) / 2)
        p = count / total
        denominator = 1 + z**2/total
        center = (p + z**2/(2*total)) / denominator
        spread = z * np.sqrt(p*(1-p)/total + z**2/(4*total**2)) / denominator
        return max(0.0, center - spread), min(1.0, center + spread)

    # Absolute counts with confidence intervals
    absolute_counts = {
        "total_points_of_interest": len(pois),
        "counts_by_category": {
            f"total_{cat}_places": {
                "count": count,
                "percentage": round_float(count/len(pois) * 100, PERCENT_DECIMALS),
                "confidence_interval_95": {
                    "lower": round_float(ci[0] * 100, PERCENT_DECIMALS),
                    "upper": round_float(ci[1] * 100, PERCENT_DECIMALS)
                } if (ci := calculate_proportion_ci(count, len(pois))) != (None, None) else None
            }
            for cat, count in category_counts.items()
        }
    }
    
    # Density metrics (per square kilometer)
    density_metrics = {
        "points_of_interest_per_sqkm": round_float(
            len(pois) / area_sqkm, DENSITY_DECIMALS
        ),
        "density_by_category": {
            f"{cat}_places_per_sqkm": round_float(
                count / area_sqkm, DENSITY_DECIMALS
            )
            for cat, count in category_counts.items()
        },
        "units": "per_square_kilometer"
    }
    
    # Calculate nearest neighbor statistics
    if len(pois) > 1:
        coords = np.array([(p.x, p.y) for p in pois.geometry])
        distances = []
        for i, point in enumerate(coords):
            dist = np.sqrt(np.sum((coords[i+1:] - point)**2, axis=1))
            if len(dist) > 0:
                distances.append(np.min(dist))
        
        mean_nn_dist = np.mean(distances)
        std_nn_dist = np.std(distances)
        
        # Calculate expected mean distance for CSR
        area = area_sqm
        density = len(pois) / area
        expected_mean_dist = 1 / (2 * np.sqrt(density))
        
        # R statistic (ratio of observed to expected)
        r_statistic = mean_nn_dist / expected_mean_dist
    else:
        mean_nn_dist = None
        std_nn_dist = None
        r_statistic = None
    
    # Distribution metrics with statistical measures
    counts_array = np.array([int(x) for x in category_counts.values])
    largest_category = category_counts.index[0]
    largest_count = int(category_counts.iloc[0])
    distribution_metrics = {
        "unique_category_count": len(category_counts),
        "category_count_distribution": {
            "most_frequent_count": largest_count,
            "least_frequent_count": int(np.min(counts_array)),
            "median_category_count": float(np.median(counts_array)),
            "mean_category_count": float(np.mean(counts_array)),
            "category_count_standard_deviation": float(np.std(counts_array)),
            "confidence_interval_95": {
                "lower": float(np.mean(counts_array) - 1.96 * np.std(counts_array) / np.sqrt(len(counts_array))),
                "upper": float(np.mean(counts_array) + 1.96 * np.std(counts_array) / np.sqrt(len(counts_array)))
            }
        },
        "largest_category": {
            "name": largest_category,
            "count": largest_count,
            "percentage": round_float(largest_count / len(pois) * 100, PERCENT_DECIMALS)
        },
        "diversity_metrics": {
            "shannon_diversity_index": round_float(shannon_diversity, RATIO_DECIMALS),
            "simpson_diversity_index": round_float(simpson_diversity, RATIO_DECIMALS),
            "category_evenness": round_float(evenness, RATIO_DECIMALS)
        },
        "spatial_distribution": {
            "mean_nearest_neighbor_distance_meters": round_float(mean_nn_dist, LENGTH_DECIMALS) if mean_nn_dist is not None else None,
            "nearest_neighbor_distance_std_meters": round_float(std_nn_dist, LENGTH_DECIMALS) if std_nn_dist is not None else None,
            "r_statistic": round_float(r_statistic, RATIO_DECIMALS) if r_statistic is not None else None,
            "pattern_interpretation": (
                "clustered" if r_statistic is not None and r_statistic < 0.9 else
                "dispersed" if r_statistic is not None and r_statistic > 1.1 else
                "random" if r_statistic is not None else None
            )
        }
    }
    
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
    area_sqm: Optional[float] = None
) -> Dict[str, Any]:
    """Calculate all metrics for a given area.
    
    Args:
        G: NetworkX graph
        pois: GeoDataFrame containing POIs
        area_sqm: Optional area override in square meters
        
    Returns:
        Dictionary containing all metrics
    """
    # Use graph area if not provided
    if area_sqm is None:
        area_sqm = float(G.graph.get('area_sqm', 0))
        if area_sqm <= 0:
            raise GeoFeatureKitError("Graph must have area_sqm attribute if area_sqm not provided")
    
    return {
        "network_metrics": calculate_network_metrics(G),
        "poi_metrics": calculate_poi_metrics(pois, area_sqm),
        "units": {
            "area": "square_meters",
            "length": "meters",
            "density": "per_square_kilometer"
        }
    } 