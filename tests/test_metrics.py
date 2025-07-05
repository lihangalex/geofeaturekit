"""Test cases for metrics calculations."""

import pytest
import numpy as np
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geofeaturekit.core.metrics import (
    calculate_network_metrics,
    calculate_poi_metrics,
    calculate_all_metrics
)
from geofeaturekit.exceptions import GeoFeatureKitError
from .utils import (
    create_test_graph,
    create_test_pois,
    calculate_expected_metrics,
    assert_metrics_match
)

class TestNetworkMetrics:
    """Test suite for network metrics calculations."""
    
    def test_empty_graph(self):
        """Test that empty graph raises error."""
        G = nx.MultiDiGraph()
        with pytest.raises(GeoFeatureKitError, match="Cannot calculate metrics for empty graph"):
            calculate_network_metrics(G)
    
    def test_grid_network(self):
        """Test metrics for perfect grid network."""
        # Create 3x3 grid (9 nodes, 24 edges because it's directed)
        G = create_test_graph(num_nodes=9, grid_layout=True, radius_meters=500)
        metrics = calculate_network_metrics(G)
        
        # Basic metrics validation
        assert metrics['basic_metrics']['total_nodes'] == 9
        assert metrics['basic_metrics']['total_street_segments'] == 24  # Each undirected edge becomes 2 directed edges
        assert metrics['basic_metrics']['total_intersections'] == 4  # Corner nodes are not intersections
        assert metrics['basic_metrics']['total_dead_ends'] == 4  # Corner nodes are dead ends
        
        # Connectivity validation
        assert metrics['connectivity_metrics']['streets_to_nodes_ratio'] == pytest.approx(24/9, rel=1e-5)
        assert metrics['connectivity_metrics']['average_connections_per_node']['value'] == pytest.approx(2.67, rel=1e-2)
        
        # Pattern validation
        pattern = metrics['street_pattern_metrics']
        assert pattern['ninety_degree_intersection_ratio'] == pytest.approx(1.0, rel=1e-2)
        assert pattern['bearing_entropy'] is not None
    
    def test_random_network(self):
        """Test metrics for random network."""
        G = create_test_graph(num_nodes=10, grid_layout=False, radius_meters=500)
        metrics = calculate_network_metrics(G)
        
        # Basic validation
        assert metrics['basic_metrics']['total_nodes'] > 0
        assert metrics['basic_metrics']['total_street_segments'] > 0
        assert metrics['basic_metrics']['total_street_length_meters'] > 0
        
        # Density validation
        assert metrics['density_metrics']['intersections_per_sqkm'] >= 0
        assert metrics['density_metrics']['street_length_per_sqkm'] > 0
    
    def test_network_statistics(self):
        """Test statistical metrics calculation."""
        G = create_test_graph(num_nodes=20, grid_layout=False, radius_meters=500)
        metrics = calculate_network_metrics(G)
        
        # Length distribution
        dist = metrics['street_pattern_metrics']['street_segment_length_distribution']
        assert dist['minimum_meters'] <= dist['maximum_meters']
        assert dist['mean_meters'] >= dist['minimum_meters']
        assert dist['mean_meters'] <= dist['maximum_meters']
        assert dist['std_dev_meters'] >= 0
        
        # Bearing entropy
        assert metrics['street_pattern_metrics']['bearing_entropy'] >= 0
        if metrics['street_pattern_metrics']['ninety_degree_intersection_ratio'] is not None:
            assert 0 <= metrics['street_pattern_metrics']['ninety_degree_intersection_ratio'] <= 1

class TestPOIMetrics:
    """Test suite for POI metrics calculations."""
    
    def test_empty_pois(self):
        """Test metrics for empty POI set."""
        pois = gpd.GeoDataFrame(columns=['amenity', 'name', 'geometry'])
        metrics = calculate_poi_metrics(pois, 1000000)
        
        assert metrics['absolute_counts']['total_points_of_interest'] == 0
        assert metrics['density_metrics']['points_of_interest_per_sqkm'] is None
    
    @pytest.mark.parametrize("pattern", ["random", "clustered", "grid"])
    def test_spatial_patterns(self, pattern):
        """Test metrics for different spatial patterns."""
        pois = create_test_pois(num_pois=100, pattern=pattern)
        metrics = calculate_poi_metrics(pois, 1000000)
        
        # Basic validation
        assert metrics['absolute_counts']['total_points_of_interest'] == 100
        assert metrics['density_metrics']['points_of_interest_per_sqkm'] > 0
        
        # Pattern validation
        dist = metrics['distribution_metrics']['spatial_distribution']
        assert dist['mean_nearest_neighbor_distance_meters'] > 0
        assert dist['nearest_neighbor_distance_std_meters'] >= 0
        assert dist['r_statistic'] > 0
        assert dist['pattern_interpretation'] in ['clustered', 'dispersed', 'random']
        
        # Check if pattern matches expectation
        if pattern == "clustered":
            assert dist['r_statistic'] < 1
            assert dist['pattern_interpretation'] == 'clustered'
        elif pattern == "grid":
            assert dist['r_statistic'] > 1
            assert dist['pattern_interpretation'] == 'dispersed'
    
    def test_diversity_metrics(self):
        """Test diversity metrics calculation."""
        pois = create_test_pois(num_pois=100)
        metrics = calculate_poi_metrics(pois, 1000000)
        
        # Diversity validation
        div = metrics['distribution_metrics']['diversity_metrics']
        assert 0 <= div['shannon_diversity_index']
        assert 0 <= div['simpson_diversity_index'] <= 1
        assert 0 <= div['category_evenness'] <= 1

class TestIntegration:
    """Integration tests for all metrics."""
    
    def test_complete_analysis(self):
        """Test complete analysis with network and POIs."""
        # Create test data
        G = create_test_graph(num_nodes=25, grid_layout=True)
        pois = create_test_pois(num_pois=100, pattern='random')
        
        # Calculate metrics
        metrics = calculate_all_metrics(G, pois, None)
        
        # Validate structure
        assert 'network_metrics' in metrics
        assert 'poi_metrics' in metrics
        assert 'units' in metrics
        
        # Validate network metrics
        net = metrics['network_metrics']
        assert net['basic_metrics']['total_nodes'] == 25
        assert net['basic_metrics']['total_street_segments'] > 0
        assert net['density_metrics']['street_length_per_sqkm'] > 0
        
        # Validate POI metrics
        poi = metrics['poi_metrics']
        assert poi['absolute_counts']['total_points_of_interest'] == 100
        assert poi['density_metrics']['points_of_interest_per_sqkm'] > 0
    
    @pytest.mark.parametrize("radius", [100, 500, 1000])
    def test_scale_invariance(self, radius):
        """Test that metrics are scale-invariant where appropriate."""
        G = create_test_graph(num_nodes=16, grid_layout=True, radius_meters=radius)
        pois = create_test_pois(num_pois=50, radius_meters=radius)
        
        metrics = calculate_all_metrics(G, pois, None)
        
        # Network metrics that should be scale-invariant
        net = metrics['network_metrics']
        assert net['basic_metrics']['total_nodes'] == 16
        # For 4x4 grid: 48 directed edges / 16 nodes = 3.0 ratio
        assert net['connectivity_metrics']['streets_to_nodes_ratio'] == pytest.approx(3.0, rel=1e-1)
        
        # POI metrics that should be scale-invariant
        poi = metrics['poi_metrics']
        assert poi['absolute_counts']['total_points_of_interest'] == 50
        
        # Density metrics should scale with area
        area_sqkm = (3.14159 * radius ** 2) / 1000000  # Area in square kilometers
        expected_poi_density = 50 / area_sqkm  # POIs per square kilometer
        actual_poi_density = poi['density_metrics']['points_of_interest_per_sqkm']
        assert actual_poi_density == pytest.approx(expected_poi_density, rel=1e-1)

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Invalid graph
    with pytest.raises(GeoFeatureKitError):
        calculate_network_metrics(nx.Graph())  # Not a MultiDiGraph
    
    # Invalid POIs
    with pytest.raises(GeoFeatureKitError):
        calculate_poi_metrics(pd.DataFrame(), 1000)  # Not a GeoDataFrame
    
    # Invalid area
    with pytest.raises(GeoFeatureKitError):
        calculate_poi_metrics(gpd.GeoDataFrame(), -1)  # Negative area 