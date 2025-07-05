"""Points of Interest (POI) utilities."""

import osmnx as ox
import time
from requests.exceptions import RequestException
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, MultiPolygon
from geofeaturekit.core.config import DEFAULT_CRS
import pandas as pd
from ..utils.progress import create_progress_bar, log_analysis_start, log_analysis_complete, log_error
import geopandas as gpd
from collections import defaultdict
from .area import calculate_area_sqkm
from .formatting import round_float, DENSITY_DECIMALS, PERCENT_DECIMALS

from .poi_categories import CATEGORY_WEIGHTS, POI_CATEGORIES

# Define meaningful POI categories
IMPORTANT_AMENITIES = {
    # Food and Drink
    'restaurant', 'cafe', 'bar', 'pub', 'fast_food',
    
    # Shopping
    'marketplace', 'supermarket', 'convenience', 'department_store',
    
    # Services
    'bank', 'pharmacy', 'post_office', 'hospital', 'clinic', 'doctors',
    
    # Transportation
    'bus_station', 'taxi', 'bicycle_rental', 'car_sharing', 'parking',
    
    # Culture and Entertainment
    'theatre', 'cinema', 'library', 'museum', 'arts_centre',
    
    # Education
    'school', 'university', 'college', 'kindergarten',
    
    # Other Important
    'police', 'fire_station', 'townhall', 'courthouse'
}

# Categories to group together
CATEGORY_GROUPS = {
    'dining': {'restaurant', 'cafe', 'bar', 'pub', 'fast_food'},
    'shopping': {'marketplace', 'supermarket', 'convenience', 'department_store'},
    'healthcare': {'hospital', 'clinic', 'doctors', 'pharmacy'},
    'transportation': {'bus_station', 'taxi', 'bicycle_rental', 'car_sharing', 'parking'},
    'culture': {'theatre', 'cinema', 'library', 'museum', 'arts_centre'},
    'education': {'school', 'university', 'college', 'kindergarten'},
    'services': {'bank', 'post_office'},
    'emergency': {'police', 'fire_station'},
    'government': {'townhall', 'courthouse'}
}

def get_poi_tags() -> dict:
    """Get POI tags dictionary for OSMnx query.
    
    Returns:
        Dict mapping OSM tag types to lists of values
    """
    tags = {}
    for category_tags in POI_CATEGORIES.values():
        for tag_type, values in category_tags.items():
            if tag_type not in tags:
                tags[tag_type] = []
            tags[tag_type].extend(values)
    return tags

def get_geometry_centroid(geom) -> Tuple[float, float]:
    """Extract centroid coordinates from a geometry object.
    
    Args:
        geom: A shapely geometry object (Point, LineString, Polygon, or MultiPolygon)
        
    Returns:
        Tuple[float, float]: (latitude, longitude) of the geometry's centroid
        
    Raises:
        ValueError: If geometry type is not supported
    """
    try:
        if hasattr(geom, 'centroid'):
            # This handles Polygon, MultiPolygon, and LineString
            centroid = geom.centroid
            return (centroid.y, centroid.x)
        elif isinstance(geom, Point):
            return (geom.y, geom.x)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geom)}")
    except Exception as e:
        raise ValueError(f"Failed to extract centroid from geometry: {str(e)}")

def check_tag_value(tag_value: Any) -> Optional[str]:
    """Check a tag value and return a valid string if found.
    
    Args:
        tag_value: The tag value to check (can be scalar or array-like)
        
    Returns:
        Optional[str]: The first valid string value, or None if no valid value found
    """
    if tag_value is None:
        return None
        
    try:
        # Handle list-like objects without converting to numpy array
        if hasattr(tag_value, '__iter__') and not isinstance(tag_value, (str, bytes)):
            # Iterate directly over the values
            for val in tag_value:
                if pd.notna(val):  # Use pandas NA check which handles all types
                    return str(val)
            return None
        else:
            # Handle scalar value
            return str(tag_value) if pd.notna(tag_value) else None
    except Exception as e:
        print(f"  Warning: Error processing tag value: {str(e)}")
        return None

def check_tag_matches(tag_value: Any, valid_values: List[str]) -> Optional[str]:
    """Check if a tag value matches any of the valid values.
    
    Args:
        tag_value: The tag value to check (can be scalar or array-like)
        valid_values: List of valid values to match against
        
    Returns:
        Optional[str]: The first matching value if found, None otherwise
    """
    if tag_value is None:
        return None
        
    # Convert valid values to set for faster lookup
    valid_set = {str(v).lower() for v in valid_values}
    
    try:
        # Handle list-like objects without converting to numpy array
        if hasattr(tag_value, '__iter__') and not isinstance(tag_value, (str, bytes)):
            # Iterate directly over the values
            for val in tag_value:
                if pd.notna(val):  # Use pandas NA check which handles all types
                    val_str = str(val).lower()
                    if val_str in valid_set:
                        return str(val)
            return None
        else:
            # Handle scalar value
            if pd.notna(tag_value):
                val_str = str(tag_value).lower()
                return str(tag_value) if val_str in valid_set else None
            return None
    except Exception as e:
        print(f"  Warning: Error matching tag value: {str(e)}")
        return None

def get_pois(
    latitude: float,
    longitude: float,
    radius_meters: int,
    categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get points of interest with detailed information.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Search radius in meters
        categories: Optional list of OSM categories to include
            If None, includes all categories
            
    Returns:
        Dictionary mapping POI categories to lists of POIs with their details
    """
    # Create point
    point = (latitude, longitude)
    
    # Get POIs from OSM
    try:
        pois = ox.features_from_point(point, dist=radius_meters, tags={'amenity': True})
    except Exception as e:
        log_error(f"{latitude}, {longitude}", e)
        return {}
    
    # Process POIs by category
    if pois is None or pois.empty:
        return {}
    
    poi_details = {}
    for _, poi in create_progress_bar(pois.iterrows(), desc="Counting POIs", total=len(pois)):
        category = poi.get('amenity', 'other')
        if categories is None or category in categories:
            if category not in poi_details:
                poi_details[category] = {
                    "count": 0,
                    "items": []
                }
            
            # Get POI coordinates
            try:
                lat, lon = get_geometry_centroid(poi.geometry)
            except ValueError:
                continue
            
            # Get POI name or generate a descriptive placeholder
            name = poi.get('name')
            if pd.isna(name) or not name:
                # Try to get a more descriptive name from other tags
                name = (
                    poi.get('operator') or 
                    poi.get('brand') or 
                    poi.get('description') or
                    f"Unnamed {category.replace('_', ' ').title()}"
                )
            
            # Extract relevant POI details
            poi_info = {
                "name": name,  # This will now be either the actual name or a descriptive placeholder
                "latitude": lat,
                "longitude": lon,
                "tags": {
                    key: check_tag_value(value)
                    for key, value in poi.items()
                    if key not in ['geometry', 'osmid', 'name'] and check_tag_value(value) is not None
                }
            }
            
            poi_details[category]["items"].append(poi_info)
            poi_details[category]["count"] += 1
    
    return poi_details

def process_pois(pois: Dict[str, Any]) -> Dict[str, Any]:
    """Process POIs into category counts and details.
    
    Args:
        pois: Dictionary of categorized POIs with detailed information
        
    Returns:
        Dict with category counts and detailed POI information
    """
    return {
        "counts": {category: data["count"] for category, data in pois.items()},
        "details": {category: data["items"] for category, data in pois.items()}
    }

def _categorize_poi(amenity: str) -> str:
    """Map a POI to its category group."""
    for category, amenities in CATEGORY_GROUPS.items():
        if amenity in amenities:
            return category
    return 'other'

def analyze_pois(pois: gpd.GeoDataFrame, area_hectares: float) -> Dict[str, Any]:
    """Analyze POIs and calculate metrics.

    Args:
        pois: GeoDataFrame containing POIs
        area_hectares: Area in hectares

    Returns:
        Dictionary containing POI metrics
    """
    if pois.empty:
        return {
            "total_pois": 0,
            "density_metrics": {
                "total_density": 0,
                "density_by_category": {},
                "units": "per_hectare"
            },
            "categories": {},
            "area_hectares": round_float(area_hectares, DENSITY_DECIMALS)
        }

    # Calculate total metrics
    total_pois = len(pois)
    
    # Calculate density metrics
    density = _calculate_density_metrics(pois, area_hectares)
    
    # Calculate category distribution
    categories = pois['category'].value_counts().to_dict()
    category_percentages = (pois['category'].value_counts(normalize=True) * 100).apply(
        lambda x: round_float(x, PERCENT_DECIMALS)
    ).to_dict()

    return {
        "total_pois": total_pois,
        "density_metrics": density,
        "categories": {
            cat: {
                "count": count,
                "percentage": category_percentages[cat]
            }
            for cat, count in categories.items()
        },
        "area_hectares": round_float(area_hectares, DENSITY_DECIMALS)
    }

def _calculate_density_metrics(
    pois: gpd.GeoDataFrame,
    area_hectares: float
) -> Dict[str, Any]:
    """Calculate POI density metrics.

    Args:
        pois: GeoDataFrame containing POIs
        area_hectares: Area in hectares

    Returns:
        Dictionary of density metrics (all per hectare)
    """
    if pois.empty or area_hectares <= 0:
        return {
            "total_density": 0,
            "density_by_category": {},
            "units": "per_hectare"
        }

    # Calculate total POIs
    total_pois = len(pois)
    
    # Calculate density metrics
    density_metrics = {
        "total_density": round_float(
            total_pois / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
        ),
        "density_by_category": {
            category: round_float(
                count / area_hectares if area_hectares > 0 else 0, DENSITY_DECIMALS
            )
            for category, count in pois['category'].value_counts().items()
        },
        "units": "per_hectare"
    }
    
    return density_metrics

def get_poi_stats(
    latitude: float,
    longitude: float,
    radius_meters: int
) -> Dict[str, Any]:
    """Get POI statistics for a location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Analysis radius in meters
        
    Returns:
        Dictionary containing:
        - poi_counts: Raw counts by category
        - density_metrics: POIs per square kilometer
        - diversity_metrics: Category distribution stats
        - amenity_scores: Weighted importance scores
    """
    # Get POIs from OSM
    tags = {
        'amenity': True,
        'leisure': True,
        'shop': True,
        'tourism': True,
        'historic': True,
        'natural': True,
        'building': ['civic', 'public', 'retail', 'commercial']
    }
    
    pois = ox.features_from_point(
        (latitude, longitude),
        tags,
        dist=radius_meters
    )
    
    if pois.empty:
        return _empty_poi_stats()
    
    # Calculate area in square kilometers
    area_sqkm = calculate_area_sqkm(radius_meters)
    
    # Categorize POIs
    categorized = _categorize_pois(pois)
    
    # Calculate density metrics
    density = _calculate_density_metrics(categorized, area_sqkm)
    
    # Calculate diversity metrics
    diversity = _calculate_diversity_metrics(categorized)
    
    # Calculate amenity scores
    scores = _calculate_amenity_scores(categorized)
    
    return {
        "poi_counts": categorized,
        "density_metrics": density,
        "diversity_metrics": diversity,
        "amenity_scores": scores
    }

def _empty_poi_stats() -> Dict[str, Any]:
    """Return empty statistics structure."""
    return {
        "poi_counts": {cat: 0 for cat in POI_CATEGORIES.keys()},
        "density_metrics": {
            "total_density_per_sqkm": 0.0,
            "category_density": {cat: 0.0 for cat in POI_CATEGORIES.keys()}
        },
        "diversity_metrics": {
            "category_count": 0,
            "simpson_diversity": 0.0,
            "evenness": 0.0,
            "dominant_category": None
        },
        "amenity_scores": {
            "total_score": 0.0,
            "category_scores": {cat: 0.0 for cat in POI_CATEGORIES.keys()}
        }
    }

def _categorize_pois(pois) -> Dict[str, int]:
    """Categorize POIs into meaningful groups.
    
    Args:
        pois: GeoDataFrame of POIs
        
    Returns:
        Dictionary mapping categories to counts
    """
    counts = defaultdict(int)
    
    for _, poi in pois.iterrows():
        # Get all tags
        tags = {k: v for k, v in poi.items() if isinstance(k, str)}
        
        # Find matching category
        category = None
        for cat, patterns in POI_CATEGORIES.items():
            if _matches_category(tags, patterns):
                category = cat
                break
        
        if category:
            counts[category] += 1
    
    # Ensure all categories are present
    return {cat: counts.get(cat, 0) for cat in POI_CATEGORIES.keys()}

def _matches_category(tags: Dict[str, str], patterns: List[Dict[str, Set[str]]]) -> bool:
    """Check if POI tags match category patterns.
    
    Args:
        tags: POI tags
        patterns: List of tag patterns that define the category
        
    Returns:
        Whether the POI matches the category
    """
    for pattern in patterns:
        matches = True
        for key, values in pattern.items():
            tag_value = str(tags.get(key, '')).lower()
            if not tag_value or tag_value not in values:
                matches = False
                break
        if matches:
            return True
    return False

def _calculate_diversity_metrics(categorized: Dict[str, int]) -> Dict[str, Any]:
    """Calculate POI diversity metrics.
    
    Args:
        categorized: Dictionary of POI counts by category
        
    Returns:
        Dictionary of diversity metrics
    """
    total_pois = sum(categorized.values())
    
    if total_pois == 0:
        return {
            "category_count": 0,
            "simpson_diversity": 0.0,
            "evenness": 0.0,
            "dominant_category": None
        }
    
    # Calculate proportions
    proportions = [count / total_pois for count in categorized.values()]
    
    # Simpson's diversity index (1 - D)
    simpson = 1 - sum(p * p for p in proportions)
    
    # Evenness (normalized Simpson's index)
    max_simpson = 1 - (1 / len(categorized))
    evenness = simpson / max_simpson if max_simpson > 0 else 0
    
    # Find dominant category
    dominant = max(categorized.items(), key=lambda x: x[1])
    
    return {
        "category_count": sum(1 for count in categorized.values() if count > 0),
        "simpson_diversity": float(simpson),
        "evenness": float(evenness),
        "dominant_category": {
            "name": dominant[0],
            "count": dominant[1],
            "percentage": round_float(dominant[1] / total_pois * 100 if total_pois > 0 else 0.0, PERCENT_DECIMALS)
        }
    }

def _calculate_amenity_scores(categorized: Dict[str, int]) -> Dict[str, Any]:
    """Calculate weighted amenity scores.
    
    Args:
        categorized: Dictionary of POI counts by category
        
    Returns:
        Dictionary of amenity scores
    """
    # Calculate weighted scores
    category_scores = {}
    total_score = 0.0
    
    for category, count in categorized.items():
        weight = CATEGORY_WEIGHTS.get(category, 1.0)
        score = count * weight
        category_scores[category] = float(score)
        total_score += score
    
    return {
        "total_score": float(total_score),
        "category_scores": category_scores
    }

def extract_pois(
    gdf: gpd.GeoDataFrame,
    categories: List[str] = None
) -> gpd.GeoDataFrame:
    """Extract and categorize points of interest.
    
    Args:
        gdf: GeoDataFrame containing POI data
        categories: List of OSM categories to include
                   If None, includes all categories
    
    Returns:
        GeoDataFrame containing filtered and processed POIs
    """
    if gdf.empty:
        return gdf
        
    # Default categories if none specified
    if categories is None:
        categories = [
            'amenity',
            'leisure',
            'shop',
            'tourism',
            'healthcare',
            'education'
        ]
    
    # Create mask for each category
    masks = []
    for category in categories:
        mask = gdf[category].notna()
        if mask.any():
            masks.append(mask)
    
    # Combine masks
    if masks:
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask |= mask
        
        # Filter GeoDataFrame
        filtered_gdf = gdf[combined_mask].copy()
    else:
        filtered_gdf = gdf.copy()
    
    # Create unified category column
    filtered_gdf['category'] = None
    for category in categories:
        mask = filtered_gdf[category].notna()
        filtered_gdf.loc[mask, 'category'] = filtered_gdf.loc[mask, category]
    
    return filtered_gdf

def download_pois(
    latitude: float,
    longitude: float,
    radius_meters: int,
    custom_tags: Optional[Dict[str, Any]] = None
) -> gpd.GeoDataFrame:
    """Download points of interest from OpenStreetMap.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Analysis radius in meters
        custom_tags: Custom OSM tags to filter POIs
        
    Returns:
        GeoDataFrame containing points of interest
    """
    # Configure OSMnx
    ox.settings.use_cache = True
    ox.settings.log_console = False
    
    # Default POI tags
    tags = {
        'amenity': True,
        'leisure': True,
        'shop': True,
        'tourism': True,
        'historic': True,
        'office': True,
        'public_transport': True,
        'healthcare': True,
        'education': True
    }
    
    # Add custom tags if provided
    if custom_tags:
        tags.update(custom_tags)
    
    # Download POIs
    pois = ox.features_from_point(
        (latitude, longitude),
        tags=tags,
        dist=radius_meters
    )
    
    # Filter to just points
    return pois[pois.geometry.type == 'Point'].copy() 