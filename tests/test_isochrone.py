"""Tests for isochrone accessibility features."""

import pytest
import numpy as np
from geofeaturekit import features_from_coordinate
from geofeaturekit.utils.isochrone import (
    calculate_isochrone_distance,
    validate_speed_config,
    get_default_speed_config,
    extract_isochrone_features
)
from geofeaturekit.exceptions.errors import GeoFeatureKitError


class TestIsochroneCalculations:
    """Test basic isochrone calculation functions."""
    
    def test_calculate_isochrone_distance(self):
        """Test distance calculation from travel time and speed."""
        # Basic calculation: 5 minutes at 5 km/h = 416.67 meters
        distance = calculate_isochrone_distance(5, 5)
        expected = (5 * 1000) / 60 * 5  # 416.67 meters
        assert abs(distance - expected) < 0.1
        
        # Test different speeds
        assert calculate_isochrone_distance(10, 15) == 2500.0  # 10min at 15km/h
        assert calculate_isochrone_distance(3, 20) == 1000.0   # 3min at 20km/h
    
    def test_default_speed_config(self):
        """Test default speed configuration."""
        config = get_default_speed_config()
        
        assert config['walk'] == 5.0
        assert config['bike'] == 15.0
        assert config['drive'] == 40.0
        assert len(config) == 3
    
    def test_validate_speed_config_valid(self):
        """Test speed configuration validation with valid inputs."""
        valid_config = {'walk': 4.5, 'bike': 18.0, 'drive': 35.0}
        
        # Should not raise any exception
        validate_speed_config(valid_config)
    
    def test_validate_speed_config_missing_keys(self):
        """Test speed configuration validation with missing keys."""
        invalid_config = {'walk': 5.0, 'bike': 15.0}  # Missing 'drive'
        
        with pytest.raises(ValueError, match="Missing 'drive' in speed_config"):
            validate_speed_config(invalid_config)
    
    def test_validate_speed_config_non_numeric(self):
        """Test speed configuration validation with non-numeric values."""
        invalid_config = {'walk': 5.0, 'bike': '15.0', 'drive': 40.0}
        
        with pytest.raises(TypeError, match="Speed for 'bike' must be numeric"):
            validate_speed_config(invalid_config)
    
    def test_validate_speed_config_negative_values(self):
        """Test speed configuration validation with negative values."""
        invalid_config = {'walk': 5.0, 'bike': -15.0, 'drive': 40.0}
        
        with pytest.raises(ValueError, match="Speed for 'bike' must be positive"):
            validate_speed_config(invalid_config)


class TestFeaturesFromCoordinate:
    """Test the main features_from_coordinate function."""
    
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Invalid latitude
        with pytest.raises(ValueError, match="Invalid latitude"):
            features_from_coordinate(lat=91.0, lon=0.0, radius_m=500)
        
        # Invalid longitude
        with pytest.raises(ValueError, match="Invalid longitude"):
            features_from_coordinate(lat=0.0, lon=181.0, radius_m=500)
    
    def test_no_analysis_type_specified(self):
        """Test error when no analysis type is specified."""
        with pytest.raises(ValueError, match="At least one analysis type must be specified"):
            features_from_coordinate(lat=40.7580, lon=-73.9855)
    
    def test_radius_analysis(self):
        """Test radius-based analysis."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            radius_m=300,
            show_progress=False
        )
        
        assert 'radius_features' in features
        assert 'network_metrics' in features['radius_features']
        assert 'poi_metrics' in features['radius_features']
    
    def test_walking_isochrone(self):
        """Test walking isochrone analysis."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=5,
            show_progress=False
        )
        
        assert 'isochrone_features_walk' in features
        
        walk_data = features['isochrone_features_walk']
        assert 'isochrone_info' in walk_data
        assert 'network_metrics' in walk_data
        assert 'poi_metrics' in walk_data
        
        info = walk_data['isochrone_info']
        assert info['mode'] == 'walk'
        assert info['travel_time_minutes'] == 5
        assert info['speed_kmh'] == 5.0  # Default speed
        assert info['area_sqm'] > 0
        assert 'calculation_method' in info
    
    def test_biking_isochrone(self):
        """Test biking isochrone analysis."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_bike=3,
            show_progress=False
        )
        
        assert 'isochrone_features_bike' in features
        
        bike_data = features['isochrone_features_bike']
        info = bike_data['isochrone_info']
        assert info['mode'] == 'bike'
        assert info['travel_time_minutes'] == 3
        assert info['speed_kmh'] == 15.0  # Default speed
    
    def test_driving_isochrone(self):
        """Test driving isochrone analysis with very small time for speed."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_drive=1,  # Very small time for faster testing
            show_progress=False
        )
        
        assert 'isochrone_features_drive' in features
        
        drive_data = features['isochrone_features_drive']
        info = drive_data['isochrone_info']
        assert info['mode'] == 'drive'
        assert info['travel_time_minutes'] == 1
        assert info['speed_kmh'] == 40.0  # Default speed
    
    def test_multi_modal_analysis(self):
        """Test multi-modal analysis with smaller travel times for speed."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=3,  # Smaller times for faster testing
            max_travel_time_min_bike=2,
            show_progress=False
        )
        
        expected_keys = [
            'isochrone_features_walk',
            'isochrone_features_bike'
        ]
        
        for key in expected_keys:
            assert key in features
            assert 'isochrone_info' in features[key]
            assert 'poi_metrics' in features[key]
    
    def test_custom_speed_config(self):
        """Test custom speed configuration."""
        custom_speeds = {'walk': 4.5, 'bike': 18.0, 'drive': 35.0}
        
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=5,
            max_travel_time_min_bike=3,
            speed_config=custom_speeds,
            show_progress=False
        )
        
        walk_info = features['isochrone_features_walk']['isochrone_info']
        bike_info = features['isochrone_features_bike']['isochrone_info']
        
        assert walk_info['speed_kmh'] == 4.5
        assert bike_info['speed_kmh'] == 18.0
    
    def test_combined_radius_and_isochrone(self):
        """Test combined radius and isochrone analysis."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            radius_m=200,  # Smaller radius for faster testing
            max_travel_time_min_walk=3,  # Smaller time for faster testing
            show_progress=False
        )
        
        assert 'radius_features' in features
        assert 'isochrone_features_walk' in features
        
        # Both should have POI metrics
        radius_pois = features['radius_features']['poi_metrics']['absolute_counts']['total_points_of_interest']
        walk_pois = features['isochrone_features_walk']['poi_metrics']['absolute_counts']['total_points_of_interest']
        
        assert isinstance(radius_pois, int)
        assert isinstance(walk_pois, int)
        assert radius_pois >= 0
        assert walk_pois >= 0
    
    def test_negative_travel_times(self):
        """Test error handling for negative travel times."""
        with pytest.raises(GeoFeatureKitError, match="Walk travel time must be positive"):
            features_from_coordinate(
                lat=40.7580,
                lon=-73.9855,
                max_travel_time_min_walk=-5,
                show_progress=False
            )
        
        with pytest.raises(GeoFeatureKitError, match="Bike travel time must be positive"):
            features_from_coordinate(
                lat=40.7580,
                lon=-73.9855,
                max_travel_time_min_bike=-3,
                show_progress=False
            )
    
    def test_invalid_speed_config(self):
        """Test error handling for invalid speed configuration."""
        invalid_config = {'walk': 5.0, 'bike': 15.0}  # Missing 'drive'
        
        with pytest.raises(ValueError):
            features_from_coordinate(
                lat=40.7580,
                lon=-73.9855,
                max_travel_time_min_walk=5,
                speed_config=invalid_config,
                show_progress=False
            )


class TestIsochroneFeatureStructure:
    """Test the structure and content of isochrone features."""
    
    def test_isochrone_info_structure(self):
        """Test isochrone info structure."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=4,
            show_progress=False
        )
        
        info = features['isochrone_features_walk']['isochrone_info']
        
        required_fields = [
            'mode', 'travel_time_minutes', 'speed_kmh', 
            'area_sqm', 'calculation_method', 'accessible_pois'
        ]
        
        for field in required_fields:
            assert field in info
        
        # Test data types
        assert isinstance(info['mode'], str)
        assert isinstance(info['travel_time_minutes'], (int, float))
        assert isinstance(info['speed_kmh'], (int, float))
        assert isinstance(info['area_sqm'], (int, float))
        assert isinstance(info['calculation_method'], str)
        assert isinstance(info['accessible_pois'], int)
    
    def test_poi_metrics_in_isochrone(self):
        """Test POI metrics structure in isochrone results."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=5,
            show_progress=False
        )
        
        poi_metrics = features['isochrone_features_walk']['poi_metrics']
        
        # Check required POI metric sections
        assert 'absolute_counts' in poi_metrics
        assert 'density_metrics' in poi_metrics
        assert 'distribution_metrics' in poi_metrics
        
        # Check total POI count
        total_pois = poi_metrics['absolute_counts']['total_points_of_interest']
        assert isinstance(total_pois, int)
        assert total_pois >= 0
    
    def test_area_scaling_with_travel_time(self):
        """Test that isochrone area scales reasonably with travel time."""
        # Test two different travel times - use smaller times for faster testing
        features_short = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=1,  # 1 minute
            show_progress=False
        )
        
        features_long = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=3,  # 3 minutes
            show_progress=False
        )
        
        area_short = features_short['isochrone_features_walk']['isochrone_info']['area_sqm']
        area_long = features_long['isochrone_features_walk']['isochrone_info']['area_sqm']
        
        # Longer travel time should result in larger area
        assert area_long > area_short
    
    def test_speed_impact_on_area(self):
        """Test that higher speeds result in larger accessible areas."""
        features_slow = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=2,  # Smaller time for faster testing
            speed_config={'walk': 3.0, 'bike': 15.0, 'drive': 40.0},
            show_progress=False
        )
        
        features_fast = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=2,  # Smaller time for faster testing
            speed_config={'walk': 7.0, 'bike': 15.0, 'drive': 40.0},
            show_progress=False
        )
        
        area_slow = features_slow['isochrone_features_walk']['isochrone_info']['area_sqm']
        area_fast = features_fast['isochrone_features_walk']['isochrone_info']['area_sqm']
        
        # Higher speed should result in larger area for same time
        assert area_fast > area_slow


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_basic_functionality_quick(self):
        """Quick test to verify isochrone distance calculation works."""
        # Test just the calculation functions without network downloads
        from geofeaturekit.utils.isochrone import calculate_isochrone_distance
        
        # 5 minutes at 5 km/h should be about 417 meters
        distance = calculate_isochrone_distance(5, 5)
        assert 400 < distance < 450
        
        # Test speed configuration validation
        config = get_default_speed_config()
        validate_speed_config(config)  # Should not raise
        
        # Test with tiny travel time to minimize network load
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=0.5,  # 30 seconds - very fast
            show_progress=False
        )
        
        assert 'isochrone_features_walk' in features
        info = features['isochrone_features_walk']['isochrone_info']
        assert info['area_sqm'] > 0
        assert info['travel_time_minutes'] == 0.5
    
    def test_small_travel_times(self):
        """Test very small travel times."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=1,  # Very small time
            show_progress=False
        )
        
        assert 'isochrone_features_walk' in features
        info = features['isochrone_features_walk']['isochrone_info']
        assert info['area_sqm'] > 0
    
    def test_large_travel_times(self):
        """Test moderately large travel times."""
        features = features_from_coordinate(
            lat=40.7580,
            lon=-73.9855,
            max_travel_time_min_walk=10,  # Moderate time (was 30)
            show_progress=False
        )
        
        assert 'isochrone_features_walk' in features
        info = features['isochrone_features_walk']['isochrone_info']
        assert info['area_sqm'] > 0
    
    def test_remote_location(self):
        """Test isochrone calculation in a remote location with limited infrastructure."""
        # Remote location in rural area  
        features = features_from_coordinate(
            lat=45.0,  
            lon=-100.0,  # Rural North Dakota
            max_travel_time_min_walk=2,  # Small time for faster testing
            show_progress=False
        )
        
        # Should still work, even if fewer POIs are found
        assert 'isochrone_features_walk' in features
        info = features['isochrone_features_walk']['isochrone_info']
        assert info['area_sqm'] > 0 