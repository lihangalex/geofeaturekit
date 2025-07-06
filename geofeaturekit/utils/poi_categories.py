"""POI category definitions and weights."""

from typing import Dict, Set, List, Union, Any, Optional

# Define meaningful POI categories with their OSM tag patterns
POI_CATEGORIES: Dict[str, List[Dict[str, Set[str]]]] = {
    "dining": [
        {"amenity": {"restaurant", "cafe", "fast_food", "bar", "pub"}},
        {"shop": {"coffee"}}
    ],
    
    "retail": [
        {"shop": {"convenience", "supermarket", "mall", "department_store"}},
        {"amenity": {"marketplace"}},
        {"building": {"retail", "commercial"}}
    ],
    
    "education": [
        {"amenity": {"school", "university", "college", "library", "kindergarten"}},
        {"building": {"school", "university", "college"}}
    ],
    
    "healthcare": [
        {"amenity": {"hospital", "clinic", "doctors", "dentist", "pharmacy"}},
        {"healthcare": {"hospital", "clinic", "doctor", "dentist"}},
        {"building": {"hospital"}}
    ],
    
    "culture": [
        {"amenity": {"theatre", "cinema", "arts_centre", "museum"}},
        {"tourism": {"museum", "gallery", "artwork"}},
        {"leisure": {"culture_centre"}},
        {"historic": {"monument", "memorial", "archaeological_site"}}
    ],
    
    "recreation": [
        {"leisure": {"park", "sports_centre", "fitness_centre", "swimming_pool", "playground", "sports_ground", "pitch", "dog_park"}},
        {"amenity": {"park", "swimming_pool"}},
        {"sport": {"fitness", "swimming", "gym"}},
        {"building": {"sports_centre", "stadium"}}
    ],
    
    "transportation": [
        {"amenity": {"bus_station", "parking", "bicycle_parking", "car_sharing", "bicycle_rental", "taxi"}},
        {"public_transport": {"station", "stop_position", "platform"}},
        {"railway": {"station", "halt", "tram_stop", "subway_entrance"}},
        {"highway": {"bus_stop"}},
        {"aeroway": {"aerodrome", "helipad"}}
    ],
    
    "public_transit": [
        {"railway": {"subway_entrance", "station", "tram_stop", "light_rail"}},
        {"public_transport": {"stop_position", "platform", "station"}},
        {"highway": {"bus_stop"}},
        {"amenity": {"bus_station", "ferry_terminal"}},
        {"route": {"subway", "bus", "tram", "ferry"}}
    ],
    
    "water_features": [
        {"natural": {"water", "bay", "beach", "coastline", "strait"}},
        {"waterway": {"river", "stream", "canal", "dock", "dam"}},
        {"leisure": {"marina", "swimming_area"}},
        {"amenity": {"fountain", "drinking_water"}},
        {"landuse": {"basin", "reservoir"}}
    ],
    
    "green_infrastructure": [
        {"natural": {"tree", "wood", "scrub", "grassland", "meadow"}},
        {"leisure": {"park", "garden", "nature_reserve", "dog_park", "playground"}},
        {"landuse": {"forest", "grass", "meadow", "orchard", "vineyard"}},
        {"amenity": {"bench", "waste_basket"}},
        {"barrier": {"hedge"}}
    ],
    
    "community": [
        {"amenity": {"community_centre", "social_facility", "place_of_worship"}},
        {"building": {"civic", "public"}},
        {"leisure": {"community_centre"}},
        {"office": {"government"}}
    ],
    
    "financial": [
        {"amenity": {"bank", "atm"}},
        {"shop": {"money_lender"}},
        {"office": {"financial"}}
    ],
    
    "accommodation": [
        {"tourism": {"hotel", "hostel", "guest_house", "apartment"}},
        {"building": {"hotel"}},
        {"amenity": {"hotel"}}
    ],
    
    "services": [
        {"shop": {"hairdresser", "laundry", "dry_cleaning"}},
        {"amenity": {"post_office", "police", "fire_station"}},
        {"office": {"insurance", "lawyer", "estate_agent"}}
    ],
    
    "utilities": [
        {"power": {"tower", "pole", "substation", "generator"}},
        {"man_made": {"tower", "water_tower", "pumping_station", "water_treatment", "wastewater_plant"}},
        {"amenity": {"charging_station", "fuel"}},
        {"utility": {"power", "water", "gas", "telecommunications"}}
    ],
    
    "safety_emergency": [
        {"amenity": {"police", "fire_station", "hospital"}},
        {"emergency": {"fire_hydrant", "defibrillator", "assembly_point", "siren"}},
        {"highway": {"emergency_access_point"}},
        {"man_made": {"surveillance", "lighthouse"}}
    ],
    
    "natural": [
        {"natural": {"water", "beach", "wood", "tree"}},
        {"leisure": {"nature_reserve", "garden"}},
        {"landuse": {"forest", "grass", "meadow"}}
    ]
}

# Define category weights for importance scoring
CATEGORY_WEIGHTS: Dict[str, float] = {
    "dining": 1.0,              # Basic amenity
    "retail": 1.0,              # Basic amenity
    "education": 1.5,           # Important community facility
    "healthcare": 1.5,          # Important community facility
    "culture": 1.2,             # Quality of life contributor
    "recreation": 1.2,          # Quality of life contributor
    "transportation": 1.3,      # Infrastructure importance
    "public_transit": 1.4,      # High-value urban infrastructure
    "water_features": 1.1,      # Environmental value
    "green_infrastructure": 1.3, # Environmental & livability
    "community": 1.4,           # Social infrastructure
    "financial": 0.8,           # Convenience service
    "accommodation": 0.7,       # Tourism/temporary use
    "services": 0.9,            # Support services
    "utilities": 1.2,           # Critical infrastructure
    "safety_emergency": 1.5,    # Safety & security
    "natural": 1.1              # Environmental value
}

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