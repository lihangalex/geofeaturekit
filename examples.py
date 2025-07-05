"""Example usage of GeoFeatureKit."""

from geofeaturekit import features_from_location
import json

def main():
    """Run example analysis of NYC locations."""
    
    # Example locations with radius
    locations = [
        {
            'name': 'times_square',
            'latitude': 40.7580,
            'longitude': -73.9855,  # Times Square
            'radius_meters': 500
        },
        {
            'name': 'central_park',
            'latitude': 40.7829,
            'longitude': -73.9654,  # Central Park
            'radius_meters': 500
        },
        {
            'name': 'grand_central',
            'latitude': 40.7527,
            'longitude': -73.9772,  # Grand Central
            'radius_meters': 500
        }
    ]
    
    # Extract features for each location
    results = {}
    for loc in locations:
        try:
            results[loc['name']] = features_from_location({
                'latitude': loc['latitude'],
                'longitude': loc['longitude'],
                'radius_meters': loc['radius_meters']
            }, show_progress=True)
        except Exception as e:
            results[loc['name']] = {'error': str(e)}
    
    # Print raw JSON output
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 