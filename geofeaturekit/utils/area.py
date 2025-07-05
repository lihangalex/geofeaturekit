"""Area calculation utilities."""

import numpy as np
from .formatting import round_float, AREA_DECIMALS

def calculate_area(G) -> float:
    """Calculate area in square meters for a graph's region.
    
    Args:
        G: NetworkX graph with 'dist' attribute containing radius in meters
        
    Returns:
        Area in square meters
    """
    radius_meters = G.graph['dist']
    # Calculate area in square meters (π * r²)
    area_sqm = np.pi * (radius_meters ** 2)
    return round_float(area_sqm, AREA_DECIMALS)

def calculate_area_sqm(radius_meters: float) -> float:
    """Calculate area in square meters for a circular region.
    
    Args:
        radius_meters: Radius in meters
        
    Returns:
        Area in square meters
    """
    # Calculate area in square meters (π * r²)
    area_sqm = np.pi * (radius_meters ** 2)
    return round_float(area_sqm, AREA_DECIMALS)

def calculate_area_sqkm(radius_meters: float) -> float:
    """Calculate area in square kilometers for a circular region.
    
    Args:
        radius_meters: Radius in meters
        
    Returns:
        Area in square kilometers (1 km² = 1,000,000 m²)
    """
    # Calculate area in square meters first (π * r²)
    area_sqm = np.pi * (radius_meters ** 2)
    # Convert to square kilometers
    area_sqkm = area_sqm / 1_000_000
    return round_float(area_sqkm, AREA_DECIMALS)

def calculate_area_hectares(radius_meters: float) -> float:
    """Calculate area in hectares for a circular region.
    
    Args:
        radius_meters: Radius in meters
        
    Returns:
        Area in hectares (1 hectare = 10,000 m²)
    """
    # Calculate area in square meters first (π * r²)
    area_sqm = np.pi * (radius_meters ** 2)
    # Convert to hectares
    area_hectares = area_sqm / 10_000
    return round_float(area_hectares, AREA_DECIMALS) 