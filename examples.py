"""Example usage of GeoFeatureKit."""

from geofeaturekit.core.extractor import GeospatialFeatureExtractor

def main():
    # Initialize extractor with basic configuration
    extractor = GeospatialFeatureExtractor(
        radius_meters=100,  # Analysis radius in meters
        output_format='json',  # Output format (json, pandas, object)
        include_metadata=True  # Include descriptions of metrics
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
            'latitude': 40.7527,  # Grand Central Terminal
            'longitude': -73.9772
        }
    ]
    
    # Extract features using absolute metrics
    results = extractor.extract_features(locations)
    
    # Print results for each location
    for i, result in enumerate(results):
        print(f"\nLocation {i+1} Analysis Results:")
        print(f"Address: {result['location']['address']}")
        
        # Area metrics
        area = result['network_stats']['area_metrics']
        print(f"\nArea Analysis:")
        print(f"- Size: {area['area_size_sq_km']:.2f} km²")
        print(f"- Radius: {area['radius_meters']} meters")
        
        # Street metrics
        streets = result['network_stats']['street_metrics']
        print(f"\nStreet Network:")
        print(f"- Total street length: {streets['total_street_length_meters']:.0f} meters")
        print(f"- Intersections: {streets['intersections_count']} ({streets['intersections_per_sq_km']:.1f} per km²)")
        print(f"- Dead ends: {streets['dead_ends_count']}")
        
        # Network metrics
        network = result['network_stats']['network_metrics']
        print(f"\nNetwork Characteristics:")
        print(f"- Average connections per intersection: {network['average_node_degree']:.1f}")
        print(f"- Total intersections: {network['total_node_count']}")
        print(f"- Street segments: {network['total_edge_count']}")
        
        # POI metrics
        pois = result['points_of_interest']['poi_metrics']
        print(f"\nPoints of Interest:")
        print(f"- Total POIs: {pois['total_count']} ({pois['pois_per_sq_km']:.1f} per km²)")
        print(f"- Unique categories: {pois['unique_categories']}")
        
        # Category breakdown
        categories = result['points_of_interest']['category_metrics']['counts_by_type']
        if categories:
            print("\nPOI Categories:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"- {category}: {count}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 