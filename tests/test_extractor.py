"""Tests for the UrbanFeatureExtractor class."""

import pytest
from unittest.mock import patch, MagicMock

from geofeaturekit.core.extractor import UrbanFeatureExtractor
from geofeaturekit.exceptions import GeoFeatureKitError
from geofeaturekit import features_from_location

def test_extractor_initialization():
    """Test basic initialization of UrbanFeatureExtractor."""
    extractor = UrbanFeatureExtractor()
    assert extractor.radius_meters == 500
    assert extractor.use_cache is True
    assert extractor.cache_dir.endswith('/cache')

def test_invalid_radius():
    """Test that invalid radius raises GeoFeatureKitError."""
    with pytest.raises(GeoFeatureKitError):
        UrbanFeatureExtractor(radius_meters=-100)

@patch('geofeaturekit.core.extractor.calculate_all_metrics')
@patch('geofeaturekit.core.extractor.download_pois')
@patch('geofeaturekit.core.extractor.download_network')
def test_single_location(mock_network, mock_poi, mock_calculate):
    """Test extracting features for a single location."""
    # Setup mocks
    mock_network.return_value = MagicMock()
    mock_poi.return_value = []
    mock_calculate.return_value = {
        'network_metrics': {
            'basic_metrics': {'total_street_length_meters': 1000},
            'density_metrics': {},
            'connectivity_metrics': {},
            'street_pattern_metrics': {}
        },
        'poi_metrics': {
            'absolute_counts': {'total_points_of_interest': 100},
            'density_metrics': {},
            'distribution_metrics': {}
        },
        'pedestrian_network': {},
        'land_use_metrics': {},
        'data_quality_metrics': {}
    }
    
    extractor = UrbanFeatureExtractor()
    features = extractor.features_from_location(40.7128, -74.0060)  # NYC
    
    assert isinstance(features, dict)
    assert 'network_metrics' in features
    assert 'poi_metrics' in features
    assert features['network_metrics']['basic_metrics']['total_street_length_meters'] == 1000

@patch('geofeaturekit.core.extractor.calculate_all_metrics')
@patch('geofeaturekit.core.extractor.download_pois')
@patch('geofeaturekit.core.extractor.download_network')
def test_batch_locations(mock_network, mock_poi, mock_calculate):
    """Test extracting features for multiple locations."""
    # Setup mocks
    mock_network.return_value = MagicMock()
    mock_poi.return_value = []
    mock_calculate.return_value = {
        'network_metrics': {
            'basic_metrics': {'total_street_length_meters': 1000},
            'density_metrics': {},
            'connectivity_metrics': {},
            'street_pattern_metrics': {}
        },
        'poi_metrics': {
            'absolute_counts': {'total_points_of_interest': 100},
            'density_metrics': {},
            'distribution_metrics': {}
        },
        'pedestrian_network': {},
        'land_use_metrics': {},
        'data_quality_metrics': {}
    }
    
    extractor = UrbanFeatureExtractor()
    locations = [
        (40.7128, -74.0060),  # NYC
        (51.5074, -0.1278),   # London
    ]
    features = extractor.features_from_location_batch(locations)
    
    assert isinstance(features, list)
    assert len(features) == len(locations)
    for feature in features:
        assert isinstance(feature, dict)
        assert 'network_metrics' in feature
        assert 'poi_metrics' in feature
        assert feature['network_metrics']['basic_metrics']['total_street_length_meters'] == 1000

@patch('geofeaturekit.core.extractor.calculate_all_metrics')
@patch('geofeaturekit.core.extractor.download_pois')
@patch('geofeaturekit.core.extractor.download_network')
def test_features_from_location_structure(mock_network, mock_poi, mock_calculate):
    """Test detailed output structure from features_from_location."""
    # Setup mocks with complete structure
    mock_network.return_value = MagicMock()
    mock_poi.return_value = []
    mock_calculate.return_value = {
        'network_metrics': {
            'basic_metrics': {'total_street_length_meters': 1000},
            'density_metrics': {'intersections_per_sqm': 0.001},
            'connectivity_metrics': {'average_connections_per_node': 3},
            'street_pattern_metrics': {'orientation_entropy': 0.8}
        },
        'poi_metrics': {
            'absolute_counts': {'total_points_of_interest': 100},
            'density_metrics': {'poi_per_sqm': 0.0001},
            'distribution_metrics': {'unique_category_count': 10}
        },
        'pedestrian_network': {
            'intersection_spacing_meters': {'median': 50.0}
        },
        'land_use_metrics': {
            'area_measurements': {'total_area_sqm': 1000000}
        },
        'data_quality_metrics': {
            'data_completeness_percentages': {'percent_network_data_complete': 100.0}
        }
    }
    
    extractor = UrbanFeatureExtractor()
    features = extractor.features_from_location(40.7128, -74.0060)  # NYC
    
    # Check main sections
    assert 'network_metrics' in features
    assert 'poi_metrics' in features
    assert 'pedestrian_network' in features
    assert 'land_use_metrics' in features
    assert 'data_quality_metrics' in features
    
    # Check network metrics structure
    net = features['network_metrics']
    assert 'basic_metrics' in net
    assert 'density_metrics' in net
    assert 'connectivity_metrics' in net
    assert 'street_pattern_metrics' in net
    
    # Check POI metrics structure
    poi = features['poi_metrics']
    assert 'absolute_counts' in poi
    assert 'density_metrics' in poi
    assert 'distribution_metrics' in poi

def test_invalid_coordinates():
    """Test handling of invalid coordinates."""
    extractor = UrbanFeatureExtractor()
    
    # Test coordinates out of range
    with pytest.raises(ValueError):
        extractor.features_from_location(91.0, 0.0)  # Invalid latitude
    
    with pytest.raises(ValueError):
        extractor.features_from_location(0.0, 181.0)  # Invalid longitude

def test_network_download_failure():
    """Test handling of network download failures."""
    with patch('geofeaturekit.core.extractor.download_network') as mock_download:
        mock_download.side_effect = Exception("Network download failed")
        
        extractor = UrbanFeatureExtractor()
        with pytest.raises(GeoFeatureKitError):
            extractor.features_from_location(40.7128, -74.0060) 