"""Points of Interest (POI) utilities."""

import osmnx as ox
import time
from requests.exceptions import RequestException
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, MultiPolygon
from geofeaturekit.core.config import POI_CATEGORIES, DEFAULT_CRS
import pandas as pd
from ..utils.progress import create_progress_bar, log_analysis_start, log_analysis_complete, log_error
import geopandas as gpd

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

def analyze_pois(
    latitude: float,
    longitude: float,
    radius_meters: int
) -> Dict[str, Any]:
    """Analyze points of interest around a location using absolute metrics.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        radius_meters: Search radius in meters
        
    Returns:
        Dictionary containing:
        - area_metrics: Analysis area information
        - poi_metrics: Raw POI counts and statistics
        - category_metrics: Detailed category information
    """
    # Get POIs using the correct OSMnx function
    tags = {'amenity': True}
    pois = ox.features_from_point(
        (latitude, longitude),
        tags=tags,
        dist=radius_meters
    )
    
    # Calculate area in square kilometers
    area_sqkm = np.pi * (radius_meters / 1000) ** 2
    
    # Area metrics
    area_metrics = {
        "area_size_sq_km": area_sqkm,
        "radius_meters": radius_meters
    }
    
    if pois.empty:
        return {
            "area_metrics": area_metrics,
            "poi_metrics": {
                "total_important_pois": 0,
                "important_pois_per_sq_km": 0,
                "unique_categories": 0
            },
            "category_metrics": {
                "by_category": {},
                "by_type": {}
            }
        }
    
    # Count POIs by category and type
    category_counts = {}  # High-level categories (dining, shopping, etc.)
    type_counts = {}      # Specific types (restaurant, cafe, etc.)
    important_poi_count = 0
    
    for _, row in pois.iterrows():
        amenity = row.get('amenity', 'other')
        if amenity in IMPORTANT_AMENITIES:
            important_poi_count += 1
            category = _categorize_poi(amenity)
            
            # Update category counts
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Update specific type counts
            type_counts[amenity] = type_counts.get(amenity, 0) + 1
    
    # Calculate metrics
    poi_metrics = {
        "total_important_pois": important_poi_count,
        "important_pois_per_sq_km": important_poi_count / area_sqkm if area_sqkm > 0 else 0,
        "unique_categories": len(set(category_counts.keys()))
    }
    
    # Category metrics
    category_metrics = {
        "by_category": category_counts,  # High-level categories
        "by_type": type_counts,         # Specific types
        "categories_present": list(category_counts.keys())
    }
    
    return {
        "area_metrics": area_metrics,
        "poi_metrics": poi_metrics,
        "category_metrics": category_metrics
    } 