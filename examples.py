"""GeoFeatureKit usage examples."""

from geofeaturekit import extract_features, extract_multimodal_features, extract_features_batch


def analyze_location(lat, lon, radius=500):
    """Analyze a location's urban characteristics."""
    features = extract_features(lat, lon, radius)
    
    return {
        'pois': features['poi_metrics']['absolute_counts']['total_points_of_interest'],
        'poi_density': features['poi_metrics']['density_metrics']['points_of_interest_per_sqkm'],
        'street_connectivity': features['network_metrics']['connectivity_metrics']['average_connections_per_node']['value'],
        'intersections': features['network_metrics']['basic_metrics']['total_intersections']
    }


def compare_locations(locations):
    """Compare multiple locations objectively."""
    results = extract_features_batch(locations)
    
    comparison = []
    for i, result in enumerate(results):
        comparison.append({
            'location': i + 1,
            'restaurants': result['poi_metrics']['absolute_counts']['counts_by_category']['total_dining_places']['count'],
            'total_pois': result['poi_metrics']['absolute_counts']['total_points_of_interest'],
            'poi_density': result['poi_metrics']['density_metrics']['points_of_interest_per_sqkm'],
            'diversity_index': result['poi_metrics']['distribution_metrics']['diversity_metrics']['shannon_diversity_index']
        })
    
    return comparison


def analyze_accessibility(lat, lon, walk_minutes=10, bike_minutes=10):
    """Compare accessibility by transport mode."""
    features = extract_multimodal_features(
        lat, lon,
        walk_time_minutes=walk_minutes,
        bike_time_minutes=bike_minutes
    )
    
    walk_pois = features['walk_features']['poi_metrics']['absolute_counts']['total_points_of_interest']
    bike_pois = features['bike_features']['poi_metrics']['absolute_counts']['total_points_of_interest']
    
    return {
        'walk_pois': walk_pois,
        'bike_pois': bike_pois,
        'bike_to_walk_ratio': bike_pois / walk_pois if walk_pois > 0 else 0
    }


def main():
    """Demonstrate API usage with objective analysis."""
    # Manhattan location
    lat, lon = 40.7580, -73.9855
    
    print("Location Analysis:")
    analysis = analyze_location(lat, lon)
    print(f"  POIs: {analysis['pois']}")
    print(f"  POI density: {analysis['poi_density']:.1f} per km²")
    print(f"  Street connectivity: {analysis['street_connectivity']:.1f} connections/node")
    
    print("\nLocation Comparison:")
    locations = [(40.7580, -73.9855, 500), (40.7829, -73.9654, 500)]
    comparison = compare_locations(locations)
    for result in comparison:
        print(f"  Location {result['location']}:")
        print(f"    Restaurants: {result['restaurants']}")
        print(f"    Shannon diversity: {result['diversity_index']:.2f}")
    
    print("\nAccessibility Comparison:")
    accessibility = analyze_accessibility(lat, lon)
    print(f"  10min walk: {accessibility['walk_pois']} POIs")
    print(f"  10min bike: {accessibility['bike_pois']} POIs")
    print(f"  Bike/walk ratio: {accessibility['bike_to_walk_ratio']:.1f}x")


if __name__ == "__main__":
    main() 