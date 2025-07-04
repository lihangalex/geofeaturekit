"""POI category utilities."""

from typing import Dict, List, Optional, Any
import numpy as np
from geofeaturekit.core.config import POI_CATEGORIES
import pandas as pd
from ..utils.progress import create_progress_bar, log_analysis_start, log_analysis_complete, log_error

def get_poi_tags() -> Dict[str, List[str]]:
    """Get all POI tags to fetch from OSM."""
    tags = {}
    for category_tags in POI_CATEGORIES.values():
        for tag_type, values in category_tags.items():
            if tag_type not in tags:
                tags[tag_type] = []
            tags[tag_type].extend(values)
    return tags

def matches_tag_values(tag_value: Any, values: List[str]) -> bool:
    """Check if a tag value matches any of the target values.
    
    Args:
        tag_value: The value to check (can be scalar or array-like)
        values: List of valid values to match against
        
    Returns:
        bool: True if there's a match, False otherwise
    """
    if tag_value is None:
        return False
        
    # Convert values to set for faster lookup
    valid_set = {str(v).lower() for v in values}
    
    try:
        # Handle list-like objects without converting to numpy array
        if hasattr(tag_value, '__iter__') and not isinstance(tag_value, (str, bytes)):
            # Iterate directly over the values
            for val in tag_value:
                if pd.notna(val) and str(val).lower() in valid_set:
                    return True
            return False
        else:
            # Handle scalar value
            return pd.notna(tag_value) and str(tag_value).lower() in valid_set
    except Exception as e:
        print(f"  Warning: Error matching tag values: {str(e)}")
        return False

def get_poi_category(tags: Dict[str, Any]) -> Optional[str]:
    """Determine POI category based on its tags.
    
    Args:
        tags: Dictionary of OSM tags
        
    Returns:
        Optional[str]: Category name if matched, None otherwise
    """
    for category, category_tags in POI_CATEGORIES.items():
        for tag_type, values in category_tags.items():
            if tag_type in tags and matches_tag_values(tags[tag_type], values):
                return category
    return None

def get_category_for_poi(poi_dict: Dict[str, Any]) -> Optional[str]:
    """Get the category for a POI based on its tags."""
    for category, category_tags in POI_CATEGORIES.items():
        for tag_type, values in category_tags.items():
            if tag_type in poi_dict and matches_tag_values(poi_dict[tag_type], values):
                return category
    return None

def process_poi_categories(
    raw_pois: Dict[str, Any],
    categories: Optional[List[str]] = None
) -> Dict[str, int]:
    """Process raw POI data into category counts.
    
    Args:
        raw_pois: Raw POI data from OSM
        categories: Optional list of categories to include
            If None, includes all categories
            
    Returns:
        Dictionary mapping categories to counts
    """
    if not raw_pois:
        return {}
    
    counts = {}
    for category in create_progress_bar(raw_pois.keys(), desc="Processing categories"):
        if categories is None or category in categories:
            try:
                count = raw_pois[category].get('count', 0)
                if count > 0:
                    counts[category] = count
            except Exception as e:
                log_error(category, e)
                continue
    
    return counts 