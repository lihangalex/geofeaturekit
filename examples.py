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
        
        # Street network structure
        streets = result['network_stats']['street_metrics']
        nodes = streets['node_types']
        print(f"\nStreet Network Structure:")
        print(f"- Total street length: {streets['total_street_length_meters']:.0f} meters")
        print(f"- Intersections: {nodes['intersections']} ({streets['intersections_per_sq_km']:.1f} per km²)")
        print(f"- Dead ends: {nodes['dead_ends']}")
        print(f"- Other nodes (bends): {nodes['other_nodes']}")
        
        # Network characteristics
        network = result['network_stats']['network_metrics']
        print(f"\nNetwork Characteristics:")
        print(f"- Average connections per intersection: {network['average_connections_per_intersection']:.1f}")
        
        net_stats = network['street_network_stats']
        print(f"- Total street segments: {net_stats['total_street_segments']}")
        if net_stats['connected_components'] > 1:
            print(f"- Network is split into {net_stats['connected_components']} separate parts")
        
        # POI analysis
        pois = result['points_of_interest']
        metrics = pois['poi_metrics']
        print(f"\nPoints of Interest:")
        print(f"- Total important amenities: {metrics['total_important_pois']} ({metrics['important_pois_per_sq_km']:.1f} per km²)")
        
        # Show categories if any exist
        categories = pois['category_metrics']
        if categories['by_category']:
            print("\nAmenities by Category:")
            for category, count in sorted(categories['by_category'].items(), key=lambda x: x[1], reverse=True):
                print(f"- {category.title()}: {count}")
            
            print("\nDetailed Amenity Types:")
            for amenity_type, count in sorted(categories['by_type'].items(), key=lambda x: x[1], reverse=True):
                print(f"- {amenity_type.replace('_', ' ').title()}: {count}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 