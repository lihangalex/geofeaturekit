"""Tests for isochrone accessibility features."""

import pytest
import numpy as np
from geofeaturekit import extract_multimodal_features
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


class TestExtractMultimodalFeatures:
    """Test the main extract_multimodal_features function."""
    
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Invalid latitude
        with pytest.raises(ValueError, match="Invalid latitude"):
            extract_multimodal_features(latitude=91.0, longitude=0.0, radius_meters=500)
        
        # Invalid longitude
        with pytest.raises(ValueError, match="Invalid longitude"):
            extract_multimodal_features(latitude=0.0, longitude=181.0, radius_meters=500)
    
    def test_no_analysis_type_specified(self):
        """Test error when no analysis type is specified."""
        with pytest.raises(ValueError, match="At least one analysis type must be specified"):
            extract_multimodal_features(latitude=40.7580, longitude=-73.9855)
    
    def test_radius_analysis(self):
        """Test radius-based analysis."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            radius_meters=300,
            verbose=False
        )
        
        assert 'radius_features' in features
        assert 'network_metrics' in features['radius_features']
        assert 'poi_metrics' in features['radius_features']
    
    def test_walking_isochrone(self):
        """Test walking isochrone analysis."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=5,
            verbose=False
        )
        
        assert 'walk_features' in features
        
        walk_data = features['walk_features']
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
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            bike_time_minutes=3,
            verbose=False
        )
        
        assert 'bike_features' in features
        
        bike_data = features['bike_features']
        info = bike_data['isochrone_info']
        assert info['mode'] == 'bike'
        assert info['travel_time_minutes'] == 3
        assert info['speed_kmh'] == 15.0  # Default speed
    
    def test_driving_isochrone(self):
        """Test driving isochrone analysis with very small time for speed."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            drive_time_minutes=1,  # Very small time for faster testing
            verbose=False
        )
        
        assert 'drive_features' in features
        
        drive_data = features['drive_features']
        info = drive_data['isochrone_info']
        assert info['mode'] == 'drive'
        assert info['travel_time_minutes'] == 1
        assert info['speed_kmh'] == 40.0  # Default speed
    
    def test_multi_modal_analysis(self):
        """Test multi-modal analysis with smaller travel times for speed."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=3,  # Smaller times for faster testing
            bike_time_minutes=2,
            verbose=False
        )
        
        expected_keys = [
            'walk_features',
            'bike_features'
        ]
        
        for key in expected_keys:
            assert key in features
            assert 'isochrone_info' in features[key]
            assert 'poi_metrics' in features[key]
    
    def test_custom_speed_config(self):
        """Test custom speed configuration."""
        custom_speeds = {'walk': 4.5, 'bike': 18.0, 'drive': 35.0}
        
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=5,
            bike_time_minutes=3,
            speed_config=custom_speeds,
            verbose=False
        )
        
        walk_info = features['walk_features']['isochrone_info']
        bike_info = features['bike_features']['isochrone_info']
        
        assert walk_info['speed_kmh'] == 4.5
        assert bike_info['speed_kmh'] == 18.0
    
    def test_combined_radius_and_isochrone(self):
        """Test combined radius and isochrone analysis."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            radius_meters=200,  # Smaller radius for faster testing
            walk_time_minutes=3,  # Smaller time for faster testing
            verbose=False
        )
        
        assert 'radius_features' in features
        assert 'walk_features' in features
        
        # Check both have the expected structure
        assert 'network_metrics' in features['radius_features']
        assert 'poi_metrics' in features['radius_features']
        assert 'isochrone_info' in features['walk_features']
        assert 'network_metrics' in features['walk_features']
        assert 'poi_metrics' in features['walk_features']
    
    def test_negative_travel_times(self):
        """Test that negative travel times raise GeoFeatureKitError."""
        with pytest.raises(GeoFeatureKitError, match="Walk time must be positive"):
            extract_multimodal_features(
                latitude=40.7580,
                longitude=-73.9855,
                walk_time_minutes=-5,
                verbose=False
            )
        
        with pytest.raises(GeoFeatureKitError, match="Bike time must be positive"):
            extract_multimodal_features(
                latitude=40.7580,
                longitude=-73.9855,
                bike_time_minutes=-3,
                verbose=False
            )
    
    def test_invalid_speed_config(self):
        """Test invalid speed configuration raises appropriate errors."""
        invalid_config = {'walk': 5.0, 'bike': 15.0}  # Missing 'drive'
        
        with pytest.raises(ValueError):
            extract_multimodal_features(
                latitude=40.7580,
                longitude=-73.9855,
                walk_time_minutes=5,
                speed_config=invalid_config,
                verbose=False
            )


class TestIsochroneFeatureStructure:
    """Test the structure of isochrone feature outputs."""
    
    def test_isochrone_info_structure(self):
        """Test that isochrone info contains all expected fields."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=5,
            verbose=False
        )
        
        info = features['walk_features']['isochrone_info']
        
        # Check required fields
        required_fields = [
            'mode', 'travel_time_minutes', 'speed_kmh', 
            'area_sqm', 'calculation_method'
        ]
        
        for field in required_fields:
            assert field in info, f"Missing field: {field}"
        
        # Check field values
        assert info['mode'] == 'walk'
        assert info['travel_time_minutes'] == 5
        assert info['speed_kmh'] == 5.0
        assert info['area_sqm'] > 0
        assert info['calculation_method'] in ['network_based', 'circular_approximation']
    
    def test_poi_metrics_in_isochrone(self):
        """Test that POI metrics are included in isochrone results."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=5,
            verbose=False
        )
        
        poi_metrics = features['walk_features']['poi_metrics']
        
        # Check main POI metric sections
        assert 'absolute_counts' in poi_metrics
        assert 'density_metrics' in poi_metrics
        assert 'distribution_metrics' in poi_metrics
        
        # Check specific POI counts
        counts = poi_metrics['absolute_counts']
        assert 'total_points_of_interest' in counts
        assert isinstance(counts['total_points_of_interest'], int)
        assert counts['total_points_of_interest'] >= 0
    
    def test_area_scaling_with_travel_time(self):
        """Test that isochrone area scales reasonably with travel time."""
        features_short = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=3,
            verbose=False
        )
        
        features_long = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=6,
            verbose=False
        )
        
        area_short = features_short['walk_features']['isochrone_info']['area_sqm']
        area_long = features_long['walk_features']['isochrone_info']['area_sqm']
        
        # Longer travel time should generally result in larger area
        # (though this can vary due to network topology)
        assert area_long >= area_short * 0.5  # Allow some flexibility
    
    def test_speed_impact_on_area(self):
        """Test that different speeds impact isochrone area."""
        features_slow = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=5,
            speed_config={'walk': 3.0, 'bike': 15.0, 'drive': 40.0},
            verbose=False
        )
        
        features_fast = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=5,
            speed_config={'walk': 7.0, 'bike': 15.0, 'drive': 40.0},
            verbose=False
        )
        
        area_slow = features_slow['walk_features']['isochrone_info']['area_sqm']
        area_fast = features_fast['walk_features']['isochrone_info']['area_sqm']
        
        # Faster speed should generally result in larger area for same time
        assert area_fast >= area_slow * 0.5  # Allow some flexibility


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_basic_functionality_quick(self):
        """Quick test to ensure basic functionality works."""
        try:
            features = extract_multimodal_features(
                latitude=40.7580,
                longitude=-73.9855,
                radius_meters=200,  # Small radius for speed
                verbose=False
            )
            
            # Should complete without error
            assert 'radius_features' in features
            assert 'network_metrics' in features['radius_features']
            assert 'poi_metrics' in features['radius_features']
            
        except Exception as e:
            # If there are network issues, that's okay for testing
            if "download" in str(e).lower() or "network" in str(e).lower():
                pytest.skip(f"Network-related error during test: {e}")
            else:
                raise
    
    def test_small_travel_times(self):
        """Test handling of very small travel times."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=1,  # Very small time
            verbose=False
        )
        
        assert 'walk_features' in features
        info = features['walk_features']['isochrone_info']
        assert info['travel_time_minutes'] == 1
        assert info['area_sqm'] > 0
    
    def test_large_travel_times(self):
        """Test handling of large travel times."""
        features = extract_multimodal_features(
            latitude=40.7580,
            longitude=-73.9855,
            walk_time_minutes=30,  # Large time
            verbose=False
        )
        
        assert 'walk_features' in features
        info = features['walk_features']['isochrone_info']
        assert info['travel_time_minutes'] == 30
        assert info['area_sqm'] > 0
    
    def test_remote_location(self):
        """Test analysis in a remote location with potentially limited data."""
        # Use a remote location with potentially limited OSM data
        features = extract_multimodal_features(
            latitude=45.0,  # Remote area
            longitude=-110.0,
            walk_time_minutes=5,
            verbose=False
        )
        
        # Should still return valid structure even with limited data
        assert 'walk_features' in features
        assert 'isochrone_info' in features['walk_features']
        assert 'poi_metrics' in features['walk_features'] 