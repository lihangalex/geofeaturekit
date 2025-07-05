"""Property-based tests for metrics calculations."""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from geofeaturekit.core.metrics import (
    calculate_network_metrics,
    calculate_poi_metrics
)
from .utils import create_test_graph, create_test_pois

# Custom strategies
@st.composite
def graph_strategy(draw):
    """Strategy for generating valid test graphs."""
    is_grid = draw(st.booleans())
    if is_grid:
        # For grid layouts, use either 9 nodes (3x3) or 16 nodes (4x4)
        num_nodes = draw(st.sampled_from([9, 16]))
    else:
        # For non-grid layouts, use any number of nodes
        num_nodes = draw(st.integers(min_value=4, max_value=50))
    
    radius = draw(st.floats(min_value=100, max_value=1000))
    return create_test_graph(num_nodes=num_nodes, grid_layout=is_grid, radius_meters=radius)

@st.composite
def pois_strategy(draw):
    """Strategy for generating valid POI datasets."""
    num_pois = draw(st.integers(min_value=0, max_value=200))
    radius = draw(st.floats(min_value=100, max_value=2000))
    pattern = draw(st.sampled_from(['random', 'clustered', 'grid']))
    return create_test_pois(num_pois, radius_meters=radius, pattern=pattern)

class TestNetworkProperties:
    """Test network property calculations."""
    
    def test_simple_grid(self):
        """Test with a simple 3x3 grid."""
        G = create_test_graph(num_nodes=9, grid_layout=True, radius_meters=500)
        metrics = calculate_network_metrics(G)
        
        # For a 3x3 grid:
        # - 9 nodes
        # - 12 undirected edges (24 directed)
        # - Streets to nodes ratio = 24/9 = 2.666...
        assert metrics['basic_metrics']['total_nodes'] == 9
        assert metrics['basic_metrics']['total_street_segments'] == 24
        assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 2.666667) < 0.000001
    
    def test_larger_grid(self):
        """Test with a 4x4 grid."""
        G = create_test_graph(num_nodes=16, grid_layout=True, radius_meters=500)
        metrics = calculate_network_metrics(G)
        
        # For a 4x4 grid:
        # - 16 nodes
        # - 24 undirected edges (48 directed)
        # - Streets to nodes ratio = 48/16 = 3.0
        assert metrics['basic_metrics']['total_nodes'] == 16
        assert metrics['basic_metrics']['total_street_segments'] == 48
        assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 3.0) < 0.000001
    
    @given(graph_strategy())
    def test_network_metric_bounds(self, G):
        """Test that network metrics are within expected bounds."""
        metrics = calculate_network_metrics(G)
        
        # Basic metrics should always be positive
        assert metrics['basic_metrics']['total_nodes'] > 0
        assert metrics['basic_metrics']['total_street_segments'] > 0
        assert metrics['basic_metrics']['total_street_length_meters'] > 0
        assert metrics['connectivity_metrics']['streets_to_nodes_ratio'] > 0
        
        # For a grid layout, we know exact ratios
        # Check if this is a grid network by looking at node types
        is_grid = all(isinstance(n, tuple) and len(n) == 2 for n in G.nodes())
        if is_grid:
            n = int(np.sqrt(G.number_of_nodes()))
            if n * n == G.number_of_nodes():  # Perfect square = grid
                if n == 3:  # 3x3 grid
                    assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 2.666667) < 0.000001
                elif n == 4:  # 4x4 grid
                    assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 3.0) < 0.000001
        else:
            # For non-grid layouts, ratio should be between 1.5 and 3.0
            # (1.5 is the minimum for a 4-node complete graph: 6 edges / 4 nodes = 1.5)
            assert 1.5 <= metrics['connectivity_metrics']['streets_to_nodes_ratio'] <= 3.0

class TestPOIProperties:
    """Property-based tests for POI metrics."""
    
    @given(pois_strategy())
    def test_poi_metric_bounds(self, pois):
        """Test that POI metrics stay within valid bounds."""
        area_sqm = np.pi * 500**2  # Fixed area for consistency
        metrics = calculate_poi_metrics(pois, area_sqm)
        
        # Count metrics must be non-negative integers
        assert metrics['absolute_counts']['total_points_of_interest'] >= 0
        for count in metrics['absolute_counts']['counts_by_category'].values():
            if isinstance(count, dict):  # New format with confidence intervals
                assert count['count'] >= 0
                assert 0 <= count['percentage'] <= 100
            else:
                assert count >= 0
        
        # Density metrics must be non-negative
        if metrics['density_metrics']['points_of_interest_per_sqkm'] is not None:
            assert metrics['density_metrics']['points_of_interest_per_sqkm'] >= 0
        
        # Diversity metrics must be within bounds
        if len(pois) > 0:
            diversity = metrics['distribution_metrics']['diversity_metrics']
            assert 0 <= diversity['simpson_diversity_index'] <= 1
            assert 0 <= diversity['category_evenness'] <= 1
    
    @given(pois_strategy())
    def test_poi_metric_consistency(self, pois):
        """Test internal consistency of POI metrics."""
        area_sqm = np.pi * 500**2
        metrics = calculate_poi_metrics(pois, area_sqm)
        
        if len(pois) > 0:
            # Total POIs should equal sum of category counts
            category_total = sum(
                count['count'] if isinstance(count, dict) else count
                for count in metrics['absolute_counts']['counts_by_category'].values()
            )
            assert category_total == metrics['absolute_counts']['total_points_of_interest']
            
            # Spatial pattern interpretation should match R-statistic
            spatial = metrics['distribution_metrics']['spatial_distribution']
            if spatial['r_statistic'] is not None:
                if spatial['r_statistic'] < 0.9:
                    assert spatial['pattern_interpretation'] == 'clustered'
                elif spatial['r_statistic'] > 1.1:
                    assert spatial['pattern_interpretation'] == 'dispersed'
                else:
                    assert spatial['pattern_interpretation'] == 'random'

class TestScaleProperties:
    """Tests for scale-invariant properties."""
    
    @given(
        st.integers(min_value=4, max_value=25),
        st.integers(min_value=10, max_value=100),
        st.floats(min_value=100, max_value=2000)
    )
    def test_scale_invariance(self, num_nodes, num_pois, radius):
        """Test that certain metrics are invariant under scaling."""
        # Create two datasets at different scales
        scale_factor = 2.0
        
        G1 = create_test_graph(num_nodes, radius_meters=radius)
        G2 = create_test_graph(num_nodes, radius_meters=radius * scale_factor)
        
        pois1 = create_test_pois(num_pois, radius_meters=radius)
        pois2 = create_test_pois(num_pois, radius_meters=radius * scale_factor)
        
        metrics1 = calculate_network_metrics(G1)
        metrics2 = calculate_network_metrics(G2)
        
        # Topological properties should be scale-invariant
        assert metrics1['basic_metrics']['total_nodes'] == metrics2['basic_metrics']['total_nodes']
        assert metrics1['basic_metrics']['total_intersections'] == metrics2['basic_metrics']['total_intersections']
        
        # Density metrics should scale with area
        area_ratio = scale_factor ** 2
        
        # Only test density ratios if both values are non-zero
        if (metrics1['density_metrics']['intersections_per_sqkm'] > 0 and 
            metrics2['density_metrics']['intersections_per_sqkm'] > 0):
            density_ratio = metrics1['density_metrics']['intersections_per_sqkm'] / \
                           metrics2['density_metrics']['intersections_per_sqkm']
            assert abs(density_ratio - area_ratio) < 0.1  # Allow for small numerical errors

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    # Test with very large radius
    G_large = create_test_graph(num_nodes=9, radius_meters=1e6)
    metrics_large = calculate_network_metrics(G_large)
    assert all(not np.isinf(v) and not np.isnan(v) 
              for v in metrics_large['basic_metrics'].values() 
              if isinstance(v, (int, float)))
    
    # Test with very small radius
    G_small = create_test_graph(num_nodes=9, radius_meters=1)
    metrics_small = calculate_network_metrics(G_small)
    assert all(not np.isinf(v) and not np.isnan(v)
              for v in metrics_small['basic_metrics'].values()
              if isinstance(v, (int, float))) 