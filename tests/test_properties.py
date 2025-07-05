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
        assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 2.666667) < 0.001
    
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
        assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 3.0) < 0.001
    
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
                    assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 2.666667) < 0.001
                elif n == 4:  # 4x4 grid
                    assert abs(metrics['connectivity_metrics']['streets_to_nodes_ratio'] - 3.0) < 0.001
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
            
            # Basic bounds check for spatial distribution
            spatial = metrics['distribution_metrics']['spatial_distribution']
            if spatial['r_statistic'] is not None:
                # R-statistic should be positive
                assert spatial['r_statistic'] > 0
                # Pattern interpretation should be one of the valid values
                assert spatial['pattern_interpretation'] in ['clustered', 'dispersed', 'random']
    
    def test_poi_spatial_pattern_interpretation(self):
        """Test spatial pattern interpretation with deterministic patterns."""
        area_sqm = np.pi * 500**2
        
        # Test clustered pattern - all POIs in one small area
        clustered_pois = create_test_pois(20, radius_meters=500, pattern='clustered')
        metrics_clustered = calculate_poi_metrics(clustered_pois, area_sqm)
        spatial_clustered = metrics_clustered['distribution_metrics']['spatial_distribution']
        
        # Test grid pattern - should be somewhat regular/random
        grid_pois = create_test_pois(25, radius_meters=500, pattern='grid')
        metrics_grid = calculate_poi_metrics(grid_pois, area_sqm)
        spatial_grid = metrics_grid['distribution_metrics']['spatial_distribution']
        
        # Test random pattern
        random_pois = create_test_pois(30, radius_meters=500, pattern='random')
        metrics_random = calculate_poi_metrics(random_pois, area_sqm)
        spatial_random = metrics_random['distribution_metrics']['spatial_distribution']
        
        # All patterns should have valid interpretations
        for spatial in [spatial_clustered, spatial_grid, spatial_random]:
            if spatial['r_statistic'] is not None:
                assert spatial['pattern_interpretation'] in ['clustered', 'dispersed', 'random']
                # Interpretation should match r_statistic value
                r_stat = spatial['r_statistic']
                interpretation = spatial['pattern_interpretation']
                if r_stat < 0.9:
                    assert interpretation == 'clustered'
                elif r_stat > 1.1:
                    assert interpretation == 'dispersed'
                else:
                    assert interpretation == 'random'

class TestScaleProperties:
    """Tests for scale-invariant properties."""
    
    def test_scale_invariance_deterministic(self):
        """Test scale invariance with deterministic grid networks."""
        # Use deterministic 3x3 grids at different scales
        scale_factor = 2.0
        
        # Create identical grid topologies at different scales
        G1 = create_test_graph(9, grid_layout=True, radius_meters=500)    # 3x3 grid, 500m radius
        G2 = create_test_graph(9, grid_layout=True, radius_meters=1000)   # 3x3 grid, 1000m radius (2x scale)
        
        metrics1 = calculate_network_metrics(G1)
        metrics2 = calculate_network_metrics(G2)
        
        # Topological properties MUST be identical for same grid layout
        assert metrics1['basic_metrics']['total_nodes'] == metrics2['basic_metrics']['total_nodes']
        assert metrics1['basic_metrics']['total_street_segments'] == metrics2['basic_metrics']['total_street_segments']
        assert metrics1['basic_metrics']['total_intersections'] == metrics2['basic_metrics']['total_intersections']
        assert metrics1['basic_metrics']['total_dead_ends'] == metrics2['basic_metrics']['total_dead_ends']
        
        # Connectivity ratios MUST be identical (topology-dependent only)
        assert abs(metrics1['connectivity_metrics']['streets_to_nodes_ratio'] - 
                  metrics2['connectivity_metrics']['streets_to_nodes_ratio']) < 0.001
        
        # Network pattern metrics should be identical for same grid
        ninety_ratio1 = metrics1['street_pattern_metrics']['ninety_degree_intersection_ratio']
        ninety_ratio2 = metrics2['street_pattern_metrics']['ninety_degree_intersection_ratio']
        assert abs(ninety_ratio1 - ninety_ratio2) < 0.001  # Grid patterns should be identical
        
        # Length metrics should scale linearly with radius
        length_ratio = metrics2['basic_metrics']['total_street_length_meters'] / \
                      metrics1['basic_metrics']['total_street_length_meters']
        assert abs(length_ratio - scale_factor) < 0.01  # Length scales with radius
        
        # Density metrics should scale with area (radius²)
        area_scale_factor = scale_factor ** 2
        if metrics1['density_metrics']['intersections_per_sqkm'] > 0:
            density_ratio = metrics1['density_metrics']['intersections_per_sqkm'] / \
                           metrics2['density_metrics']['intersections_per_sqkm']
            assert abs(density_ratio - area_scale_factor) < 0.01  # Density scales with area
    
    @given(st.sampled_from([9, 16, 25]))  # Perfect squares for grids (3x3, 4x4, 5x5)
    def test_grid_scaling_properties(self, num_nodes):
        """Test that grid networks have predictable scaling properties."""
        # Test with different grid sizes but same scaling factor
        scale_factor = 1.5
        
        G1 = create_test_graph(num_nodes, grid_layout=True, radius_meters=400)
        G2 = create_test_graph(num_nodes, grid_layout=True, radius_meters=600)  # 1.5x scale
        
        metrics1 = calculate_network_metrics(G1)
        metrics2 = calculate_network_metrics(G2)
        
        # All topological properties must be identical
        assert metrics1['basic_metrics']['total_nodes'] == metrics2['basic_metrics']['total_nodes']
        assert metrics1['basic_metrics']['total_intersections'] == metrics2['basic_metrics']['total_intersections']
        
        # Grid patterns should be identical regardless of scale
        assert metrics1['street_pattern_metrics']['ninety_degree_intersection_ratio'] == \
               metrics2['street_pattern_metrics']['ninety_degree_intersection_ratio']

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