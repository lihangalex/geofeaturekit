"""Tests for the GeospatialFeatureExtractor class."""

import pytest
from unittest.mock import patch, MagicMock

from geofeaturekit.core.extractor import GeospatialFeatureExtractor
from geofeaturekit.core.config import AnalysisConfig
from geofeaturekit.exceptions.errors import GeoFeatureKitError

def test_extractor_initialization():
    """Test that extractor initializes correctly."""
    config = AnalysisConfig(radius_meters=300)
    extractor = GeospatialFeatureExtractor(config=config)
    assert extractor.config.radius_meters == 300

def test_invalid_radius():
    """Test that invalid radius raises ValueError."""
    with pytest.raises(ValueError):
        AnalysisConfig(radius_meters=-100)

def test_single_location():
    """Test extraction for a single location."""
    config = AnalysisConfig(radius_meters=300)
    extractor = GeospatialFeatureExtractor(
        config=config,
        output_format='object'  # Explicitly request object output
    )
    
    location = {
        "latitude": 40.7829,
        "longitude": -73.9654  # Central Park
    }
    
    result = extractor.extract_features(location)
    assert result is not None
    assert result.location.latitude == 40.7829
    assert result.location.longitude == -73.9654
    assert result.radius == 300
    assert result.network_stats is not None
    assert result.points_of_interest is not None

def test_batch_locations():
    """Test extraction for multiple locations."""
    config = AnalysisConfig(radius_meters=300)
    extractor = GeospatialFeatureExtractor(
        config=config,
        enable_embeddings=True,
        output_format='object'  # Explicitly request object output
    )
    
    locations = [
        {
            "latitude": 40.7829,
            "longitude": -73.9654  # Central Park
        },
        {
            "latitude": 40.7580,
            "longitude": -73.9855  # Times Square
        }
    ]
    
    results = extractor.extract_features(locations)
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert result.radius == 300
        assert result.network_stats is not None
        assert result.points_of_interest is not None

def test_output_formats():
    """Test different output formats."""
    config = AnalysisConfig(radius_meters=300)
    location = {
        "latitude": 40.7829,
        "longitude": -73.9654
    }
    
    # Test JSON output
    extractor_json = GeospatialFeatureExtractor(
        config=config,
        output_format='json'
    )
    result_json = extractor_json.extract_features(location)
    assert isinstance(result_json, dict)
    
    # Test pandas output
    extractor_pandas = GeospatialFeatureExtractor(
        config=config,
        output_format='pandas'
    )
    result_pandas = extractor_pandas.extract_features(location)
    assert 'latitude' in result_pandas.columns
    assert 'longitude' in result_pandas.columns 