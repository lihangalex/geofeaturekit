# GeoFeatureKit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/geofeaturekit.svg)](https://pypi.org/project/geofeaturekit/)
[![PyPI downloads](https://img.shields.io/pypi/dm/geofeaturekit.svg)](https://pypi.org/project/geofeaturekit/)
[![Tests](https://github.com/lihangalex/geofeaturekit/workflows/Test/badge.svg)](https://github.com/lihangalex/geofeaturekit/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Extract comprehensive geospatial features from coordinates for ML, urban planning, and location intelligence.**

## 🚀 Quick Start

### Installation
```bash
pip install geofeaturekit
```

### Basic Usage
```python
from geofeaturekit import extract_features

# Extract features from Times Square (quiet by default)
features = extract_features(40.7580, -73.9855, 500)  # lat, lon, radius_meters

print(f"POIs found: {features['poi_metrics']['absolute_counts']['total_points_of_interest']}")
print(f"Street length: {features['network_metrics']['basic_metrics']['total_street_length_meters']:.0f}m")
```

### Verbose Analysis
```python
# Enable detailed progress output
features = extract_features(40.7580, -73.9855, 500, verbose=True)
```

### Multi-modal Accessibility
```python
from geofeaturekit import extract_multimodal_features

# Compare walking vs biking accessibility
features = extract_multimodal_features(
    40.7580, -73.9855,
    walk_time_minutes=10,
    bike_time_minutes=5
)

walk_pois = features['walk_features']['poi_metrics']['absolute_counts']['total_points_of_interest']
bike_pois = features['bike_features']['poi_metrics']['absolute_counts']['total_points_of_interest']

print(f"10min walk: {walk_pois} POIs accessible")
print(f"5min bike: {bike_pois} POIs accessible")
```

## 🎯 What You Get

**Input:** Latitude, longitude, and radius/time  
**Output:** Comprehensive geospatial intelligence:

- **🏙️ Urban Features**: 23 comprehensive POI categories with density and diversity metrics
- **🚶 Multi-modal Accessibility**: Walking, biking, and driving isochrone analysis
- **🛣️ Street Network Analysis**: Connectivity, density, and pattern metrics
- **📊 Spatial Intelligence**: Diversity indices, clustering patterns, and distribution analysis

## 📊 Key Features

### 🏪 **23 POI Categories**
Complete coverage of urban amenities including dining, retail, healthcare, education, transportation, public transit, green infrastructure, water features, financial services, accommodation, community spaces, and more.

### 🚶 **Multi-modal Accessibility**
- **Walking**: 5.0 km/h (pedestrian accessibility)
- **Biking**: 15.0 km/h (cycling infrastructure)
- **Driving**: 40.0 km/h (car accessibility)
- **Custom speeds**: Configurable for different scenarios

### 🛣️ **Street Network Intelligence**
- **Connectivity**: Average connections per intersection
- **Density**: Street length per square kilometer
- **Pattern Analysis**: Bearing entropy and grid patterns
- **Walkability**: Pedestrian-friendly network metrics

### 📈 **Spatial Analysis**
- **Diversity Metrics**: Shannon and Simpson indices
- **Clustering Patterns**: Spatial distribution analysis
- **Density Mapping**: POI concentration by category
- **Accessibility Comparison**: Multi-modal coverage analysis

## 🌟 Clean API Features

| **Feature** | **Benefit** |
|-------------|-------------|
| ✅ **Quiet by Default** | No unexpected console output |
| ✅ **Standard Parameters** | `verbose` instead of `show_progress` |
| ✅ **Progress Callbacks** | Custom progress tracking |
| ✅ **Type Safety** | Full type hints and validation |
| ✅ **Error Handling** | Graceful failure with helpful messages |

## 🔧 Advanced Usage

### Batch Processing
```python
from geofeaturekit import extract_features_batch

# Process multiple locations efficiently
locations = [
    (40.7580, -73.9855, 500),  # Times Square
    (40.7829, -73.9654, 500),  # Central Park
    (40.7527, -73.9772, 500),  # Grand Central
]

results = extract_features_batch(locations)
```

### Custom Progress Tracking
```python
def progress_handler(message, progress):
    print(f"[{progress:.0%}] {message}")

features = extract_features(
    40.7580, -73.9855, 500,
    progress_callback=progress_handler
)
```

### Combined Analysis
```python
# Compare radius-based vs time-based accessibility
features = extract_multimodal_features(
    40.7580, -73.9855,
    radius_meters=400,        # 400m radius
    walk_time_minutes=5       # 5min walk
)

radius_pois = features['radius_features']['poi_metrics']['absolute_counts']['total_points_of_interest']
walk_pois = features['walk_features']['poi_metrics']['absolute_counts']['total_points_of_interest']

print(f"Radius (400m): {radius_pois} POIs")
print(f"Walk (5min): {walk_pois} POIs")
```

## 📝 Example Output

**Times Square Analysis (500m radius):**
```python
{
  "network_metrics": {
    "basic_metrics": {
      "total_street_length_meters": 80044.7,
      "total_intersections": 731,
      "total_nodes": 777
    },
    "connectivity_metrics": {
      "average_connections_per_node": 5.954,
      "streets_to_nodes_ratio": 1.488
    },
    "street_pattern_metrics": {
      "bearing_entropy": 2.056,
      "mean_segment_length_meters": 34.6
    }
  },
  "poi_metrics": {
    "absolute_counts": {
      "total_points_of_interest": 1076,
      "counts_by_category": {
        "total_dining_places": {"count": 400, "percentage": 37.17},
        "total_transportation_places": {"count": 190, "percentage": 17.66},
        "total_retail_places": {"count": 126, "percentage": 11.71},
        "total_public_transit_places": {"count": 96, "percentage": 8.92}
      }
    },
    "diversity_metrics": {
      "shannon_diversity_index": 2.11,
      "simpson_diversity_index": 0.81
    },
    "spatial_distribution": {
      "mean_nearest_neighbor_distance_meters": 13.2,
      "pattern_interpretation": "random"
    }
  }
}
```

## 🔬 Use Cases

### Machine Learning Feature Engineering
```python
# Generate features for price prediction
import pandas as pd

properties = pd.read_csv('real_estate.csv')
features_list = []

for _, row in properties.iterrows():
    location_features = extract_features(row['lat'], row['lon'], 1000)
    
    features_list.append({
        'poi_density': location_features['poi_metrics']['density_metrics']['points_of_interest_per_sqkm'],
                 'street_connectivity': location_features['network_metrics']['connectivity_metrics']['average_connections_per_node']['value'],
        'diversity_index': location_features['poi_metrics']['distribution_metrics']['diversity_metrics']['shannon_diversity_index']
    })

# Add to ML pipeline
features_df = pd.DataFrame(features_list)
```

### Urban Planning Analysis
```python
# Walkability assessment
features = extract_features(40.7580, -73.9855, 800, verbose=True)

walkability_score = (
    features['poi_metrics']['density_metrics']['points_of_interest_per_sqkm'] * 0.4 +
    features['network_metrics']['connectivity_metrics']['average_connections_per_node']['value'] * 100 * 0.6
)

print(f"Walkability score: {walkability_score:.1f}")
```

### Accessibility Research
```python
# Multi-modal accessibility comparison
features = extract_multimodal_features(
    40.7580, -73.9855,
    walk_time_minutes=15,
    bike_time_minutes=10,
    drive_time_minutes=5
)

for mode in ['walk', 'bike', 'drive']:
    key = f"{mode}_features"
    if key in features:
        pois = features[key]['poi_metrics']['absolute_counts']['total_points_of_interest']
        time_min = features[key]['isochrone_info']['travel_time_minutes']
        print(f"{mode.title()} ({time_min}min): {pois} POIs accessible")
```

## 🌍 Data Quality

- **International System of Units (SI)**: All measurements in meters, square kilometers
- **Confidence Intervals**: Statistical uncertainty quantification
- **Reproducible Results**: Deterministic with caching
- **Comprehensive Testing**: 57 test cases with property-based validation

## 🛠️ Installation & Requirements

```bash
pip install geofeaturekit
```

**Requirements:**
- Python 3.9+
- Automatic dependency management
- Works on Windows, macOS, and Linux

## 📚 API Reference

### Core Functions

#### `extract_features(latitude, longitude, radius_meters, *, verbose=False, cache=True, progress_callback=None)`
Extract comprehensive urban features within a circular radius.

#### `extract_multimodal_features(latitude, longitude, *, radius_meters=None, walk_time_minutes=None, bike_time_minutes=None, drive_time_minutes=None, verbose=False, cache=True, progress_callback=None)`
Multi-modal accessibility analysis with isochrone support.

#### `extract_features_batch(locations, *, verbose=False, cache=True, progress_callback=None)`
Process multiple locations efficiently.

### Parameters

- **`latitude`**: Location latitude (-90 to 90)
- **`longitude`**: Location longitude (-180 to 180)
- **`radius_meters`**: Analysis radius in meters
- **`verbose`**: Enable detailed progress output (default: False)
- **`cache`**: Use caching for faster repeated analysis (default: True)
- **`progress_callback`**: Custom progress tracking function

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with [OSMnx](https://github.com/gboeing/osmnx), [NetworkX](https://github.com/networkx/networkx), and [GeoPandas](https://github.com/geopandas/geopandas). Data © [OpenStreetMap](https://www.openstreetmap.org/) contributors.

---

**Ready to analyze any location? Start with `pip install geofeaturekit` and explore geospatial patterns! 🌍** 