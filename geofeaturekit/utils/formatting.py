"""Formatting constants for consistent decimal places."""

from typing import Optional

# Number of decimal places for different metric types
LENGTH_DECIMALS = 1  # For lengths in meters
DENSITY_DECIMALS = 2  # For density metrics (per square meter)
RATIO_DECIMALS = 3  # For ratios and normalized values
ANGLE_DECIMALS = 1  # For angles in degrees
PERCENT_DECIMALS = 2  # For percentages
AREA_DECIMALS = 2  # For areas in square meters (increased from 1 for better precision)

def round_float(value: Optional[float], decimals: int) -> Optional[float]:
    """Round a float value to specified number of decimal places.
    
    Args:
        value: Float value to round
        decimals: Number of decimal places
        
    Returns:
        Rounded float value
    """
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None 