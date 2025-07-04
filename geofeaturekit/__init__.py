"""
GeoFeatureKit - A package for extracting geospatial features from OpenStreetMap data.
"""

from geofeaturekit.core.extractor import GeospatialFeatureExtractor
from geofeaturekit.core.config import Config
from geofeaturekit.core.models import AnalysisResults

__version__ = "0.1.0"
__all__ = ['GeospatialFeatureExtractor', 'Config', 'AnalysisResults'] 