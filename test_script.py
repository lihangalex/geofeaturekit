"""Test script for the geofeaturekit package."""

from geofeaturekit.core.extractor import GeospatialFeatureExtractor
from geofeaturekit.core.config import AnalysisConfig
import json
from datetime import datetime

def run_basic_test():
    """Run basic feature extraction test."""
    # Create extractor with default settings
    config = AnalysisConfig(radius_meters=300)
    extractor = GeospatialFeatureExtractor(
        enable_embeddings=True,
        embedding_dims=64,
        output_format='json',
        config=config
    )
    
    # Test single location
    print("\nTesting single location analysis...")
    result = extractor.extract_features({
        'latitude': 40.7829,
        'longitude': -73.9654  # Central Park
    })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'output/location_analysis_{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=2)

def run_ml_feature_test():
    """Test ML feature extraction."""
    # Create extractor with ML settings
    config = AnalysisConfig(radius_meters=300)
    extractor = GeospatialFeatureExtractor(
        enable_embeddings=True,
        embedding_dims=128,
        output_format='json',
        config=config
    )
    
    # Test single location
    print("\nTesting ML feature extraction...")
    result = extractor.extract_features({
        'latitude': 40.7829,
        'longitude': -73.9654  # Central Park
    })
    
    # Save basic features
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'output/ml_features_basic_{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    # Test with reduced dimensionality
    config = AnalysisConfig(radius_meters=300)
    extractor = GeospatialFeatureExtractor(
        enable_embeddings=True,
        embedding_dims=128,
        embedding_reduce_dims=32,
        output_format='json',
        config=config
    )
    
    result = extractor.extract_features({
        'latitude': 40.7829,
        'longitude': -73.9654  # Central Park
    })
    
    # Save reduced features
    with open(f'output/ml_features_reduced_{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=2)

def run_urban_metrics_test():
    """Test urban metrics computation."""
    # Create extractor with urban metrics enabled
    config = AnalysisConfig(radius_meters=300)
    extractor = GeospatialFeatureExtractor(
        enable_urban_metrics=True,
        output_format='json',
        config=config
    )
    
    # Test single location
    print("\nTesting urban metrics computation...")
    result = extractor.extract_features({
        'latitude': 40.7829,
        'longitude': -73.9654  # Central Park
    })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'output/urban_metrics_{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=2)

def run_radius_test():
    """Test different analysis radii."""
    # Test locations
    locations = [
        {
            'name': 'Central Park',
            'latitude': 40.7829,
            'longitude': -73.9654
        }
    ]
    
    # Test different radii
    radii = [150, 300]  # meters
    
    print("\nTesting different analysis radii...")
    for radius in radii:
        # Create extractor with specific radius
        config = AnalysisConfig(radius_meters=radius)
        extractor = GeospatialFeatureExtractor(
            enable_embeddings=True,
            enable_urban_metrics=True,
            config=config
        )
        
        # Run analysis
        result = extractor.extract_features({
            'latitude': locations[0]['latitude'],
            'longitude': locations[0]['longitude']
        })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'output/radius_test_central_park_{radius}m_{timestamp}.json', 'w') as f:
            json.dump(result, f, indent=2)

def run_batch_test():
    """Test batch processing with contextual embeddings."""
    # Test locations
    locations = [
        {
            'name': 'Central Park',
            'latitude': 40.7829,
            'longitude': -73.9654
        },
        {
            'name': 'Times Square',
            'latitude': 40.7580,
            'longitude': -73.9855
        },
        {
            'name': 'Brooklyn Bridge',
            'latitude': 40.7061,
            'longitude': -73.9969
        }
    ]
    
    # Create extractor with contextual embeddings
    config = AnalysisConfig(radius_meters=300)  # Use same radius for all locations
    extractor = GeospatialFeatureExtractor(
        enable_embeddings=True,
        enable_urban_metrics=True,
        config=config
    )
    
    print("\nTesting batch processing with contextual embeddings...")
    results = extractor.extract_features([
        {'latitude': loc['latitude'], 'longitude': loc['longitude']}
        for loc in locations
    ])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'output/batch_test_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    # Run all tests
    run_basic_test()
    run_ml_feature_test()
    run_urban_metrics_test()
    run_radius_test()
    run_batch_test() 