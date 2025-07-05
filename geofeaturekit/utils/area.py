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
    
    # Use more precision for small areas to prevent precision loss
    if area_sqm < 1000:  # Less than 1000 m²
        return round_float(area_sqm, 3)  # 3 decimal places for small areas
    else:
        return round_float(area_sqm, AREA_DECIMALS)  # Standard precision for larger areas

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
    
    # Use more decimal places for small areas to prevent rounding to zero
    if area_sqkm < 0.01:
        return round_float(area_sqkm, 6)  # 6 decimal places for small areas
    else:
        return round_float(area_sqkm, AREA_DECIMALS)  # Standard precision for larger areas

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
    
    # Use more decimal places for small areas to prevent rounding to zero
    if area_hectares < 0.1:
        return round_float(area_hectares, 4)  # 4 decimal places for small areas
    else:
        return round_float(area_hectares, AREA_DECIMALS)  # Standard precision for larger areas 