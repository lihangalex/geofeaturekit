"""Example usage of GeoFeatureKit."""

from geofeaturekit.core.extractor import GeospatialFeatureExtractor

def main():
    # Initialize extractor with all parameters configured at once
    extractor = GeospatialFeatureExtractor(
        # Analysis settings
        radius_meters=100,  # default: 100
        
        # Feature flags
        enable_embeddings=True,     # default: False
        enable_urban_metrics=True,  # default: False
        enable_spectral=True,      # default: False
        
        # Embedding parameters
        embedding_dims=128,         # default: 128
        embedding_reduce_dims=64,   # default: None
        embedding_walks=200,        # default: 200
        embedding_walk_length=30,   # default: 30
        
        # Output settings
        output_format='json',       # default: 'json' (options: 'json', 'pandas', 'object')
        scale_features=True,        # default: False
        include_metadata=True       # default: True
    )
    
    # Analyze multiple locations
    locations = [
        {
            'latitude': 40.7580,  # Times Square
            'longitude': -73.9855
        },
        {
            'latitude': 40.7829,  # Central Park
            'longitude': -73.9654
        },
        {
            'latitude': 40.7527,  # Empire State Building
            'longitude': -73.9772
        }
    ]
    
    results = extractor.extract_features(locations)
    
    # Print results for each location
    for i, result in enumerate(results):
        print(f"\nLocation {i+1} Analysis Results:")
        print(result)

if __name__ == "__main__":
    main() 