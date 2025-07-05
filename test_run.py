"""Test script to demonstrate GeoFeatureKit functionality."""

from geofeaturekit import features_from_location
from pprint import pprint

def main():
    # Times Square location
    location = {
        "latitude": 40.758,
        "longitude": -73.9855,
        "radius_meters": 500  # 500m radius around the point
    }
    
    print("Extracting features for Times Square...")
    features = features_from_location(location)
    
    print("\nNetwork Metrics:")
    pprint(features['network_metrics'])
    
    print("\nPOI Metrics:")
    pprint(features['poi_metrics'])
    
    print("\nPedestrian Network:")
    pprint(features['pedestrian_network'])
    
    print("\nLand Use Metrics:")
    pprint(features['land_use_metrics'])
    
    print("\nData Quality Metrics:")
    pprint(features['data_quality_metrics'])

if __name__ == "__main__":
    main() 