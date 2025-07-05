"""Formatting constants for consistent decimal places."""

# Number of decimal places for different metric types
LENGTH_DECIMALS = 1  # For lengths in meters
DENSITY_DECIMALS = 6  # For density metrics (per square meter)
RATIO_DECIMALS = 3  # For ratios and normalized values
ANGLE_DECIMALS = 1  # For angles in degrees
PERCENT_DECIMALS = 1  # For percentages
AREA_DECIMALS = 1  # For areas in square meters

def round_float(value: float, decimals: int) -> float:
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