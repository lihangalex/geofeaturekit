# GeoFeatureKit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/geofeaturekit.svg)](https://pypi.org/project/geofeaturekit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Turn any location into rich urban data. Get POIs, street networks, and accessibility metrics from just coordinates.**

## 🚀 Quick Start

```bash
pip install geofeaturekit
```

### Basic Analysis

```python
from geofeaturekit import extract_features

# Analyze any location (lat, lon, radius_meters)
features = extract_features(40.7580, -73.9855, 500)

# Access the structured data
print(features['poi_metrics']['absolute_counts']['total_points_of_interest'])  # 1076
print(features['poi_metrics']['density_metrics']['points_of_interest_per_sqkm'])  # 1370.0
print(features['network_metrics']['basic_metrics']['total_intersections'])  # 731
```

**Sample Output Structure:**
```json
{
  "poi_metrics": {
    "absolute_counts": {
      "total_points_of_interest": 1076,
      "counts_by_category": {
        "total_dining_places": {"count": 400, "percentage": 37.17},
        "total_retail_places": {"count": 126, "percentage": 11.71},
        "total_public_transit_places": {"count": 96, "percentage": 8.92},
        "total_healthcare_places": {"count": 13, "percentage": 1.21}
      }
    },
    "density_metrics": {
      "points_of_interest_per_sqkm": 1370.0
    },
    "distribution_metrics": {
      "diversity_metrics": {"shannon_diversity_index": 2.147},
      "spatial_distribution": {"pattern_interpretation": "random"}
    }
  },
  "network_metrics": {
    "basic_metrics": {
      "total_intersections": 731,
      "total_street_length_meters": 80044.7
    },
    "connectivity_metrics": {
      "average_connections_per_node": {
        "value": 5.954,
        "confidence_interval_95": {"lower": 5.837, "upper": 6.071}
      }
    }
  }
}
```

### Multi-Modal Accessibility

```python
from geofeaturekit import extract_multimodal_features

# Compare walking vs biking reach
features = extract_multimodal_features(
    40.7580, -73.9855,
    walk_time_minutes=10,
    bike_time_minutes=10
)

walk_pois = features['walk_features']['poi_metrics']['absolute_counts']['total_points_of_interest']
bike_pois = features['bike_features']['poi_metrics']['absolute_counts']['total_points_of_interest']

print(f"🚶 10-min walk: {walk_pois} POIs reachable")     # 2104 POIs
print(f"🚴 10-min bike: {bike_pois} POIs reachable")     # 12200 POIs  
print(f"🎯 Bike advantage: {bike_pois/walk_pois:.1f}x more places")  # 5.8x
```

### Batch Processing

```python
from geofeaturekit import extract_features_batch

# Analyze multiple locations at once
locations = [
    (40.7580, -73.9855, 500),  # Times Square
    (40.7829, -73.9654, 500),  # Central Park
    (40.7527, -73.9772, 500),  # Grand Central
]

results = extract_features_batch(locations)

for i, result in enumerate(results):
    pois = result['poi_metrics']['absolute_counts']['total_points_of_interest']
    restaurants = result['poi_metrics']['absolute_counts']['counts_by_category']['total_dining_places']['count']
    print(f"Location {i+1}: {pois} POIs, {restaurants} restaurants")
    # Location 1: 1076 POIs, 400 restaurants
    # Location 2: 185 POIs, 12 restaurants  
    # Location 3: 1131 POIs, 323 restaurants
```

### Progress Tracking

```python
def progress_handler(message, progress):
    print(f"[{progress:.0%}] {message}")

# Add progress callback for long-running analysis
features = extract_features(
    40.7580, -73.9855, 1000,
    verbose=True,
    progress_callback=progress_handler
)
```

## 📊 What You Get

**POI Analysis:**
- `poi_metrics['absolute_counts']` - Raw counts by category (23 categories)
- `poi_metrics['density_metrics']` - POIs per km², density by category
- `poi_metrics['distribution_metrics']` - Diversity indices, spatial patterns

**Street Network:**
- `network_metrics['basic_metrics']` - Nodes, intersections, street length
- `network_metrics['connectivity_metrics']` - Connections per node, ratios
- `network_metrics['street_pattern_metrics']` - Bearing entropy, grid patterns

**Multi-Modal Features:**
- `walk_features`, `bike_features`, `drive_features` - Same structure as above
- `radius_features` - Circular analysis results

## 🎯 Use Cases

- **Real estate analysis**: Compare neighborhood walkability
- **Urban planning**: Assess transit accessibility  
- **Machine learning**: Generate location features for models
- **Market research**: Analyze competitor density

## 🗺️ Roadmap

- **Temporal pattern mining** - Discover how neighborhoods change across time scales  
- **Gentrification prediction** - ML models to forecast neighborhood change  
- **Causal inference engine** - Identify what actually drives urban outcomes  
- **Anomaly detection** - Identify unusual urban patterns and opportunities

*For feature requests, please contact: lihangalex@pm.me*

## 📚 API Reference

```python
# Basic analysis
extract_features(lat, lon, radius_meters, verbose=False, cache=True, progress_callback=None)

# Multi-modal accessibility  
extract_multimodal_features(lat, lon, walk_time_minutes=None, bike_time_minutes=None, drive_time_minutes=None, verbose=False, cache=True, progress_callback=None)

# Batch processing
extract_features_batch(locations, verbose=False, cache=True, progress_callback=None)
```

## 🤝 Contributing

Contributions welcome! See [Contributing Guide](CONTRIBUTING.md).

## ⚖️ License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with [OSMnx](https://github.com/gboeing/osmnx), [NetworkX](https://github.com/networkx/networkx), and [GeoPandas](https://github.com/geopandas/geopandas). Data © [OpenStreetMap](https://www.openstreetmap.org/) contributors.

---

**Ready to analyze any location? Start with `pip install geofeaturekit`** 🌍 