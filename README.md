# GeoFeatureKit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/geofeaturekit.svg)](https://pypi.org/project/geofeaturekit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/lihangalex/geofeaturekit)

**GeoFeatureKit turns raw coordinates into rich, structured geospatial features – instantly.**

## 🎯 What You Get

**Input:** Just latitude and longitude coordinates  
**Output:** Comprehensive geospatial intelligence including:

- **40+ POI categories**: restaurants, hospitals, subway stations, benches, toilets, and more
- **Street network metrics**: connectivity, total street length, segment distributions, pattern entropy
- **Spatial intelligence**: POI diversity indices (Shannon, Simpson) and clustering patterns

## 🚀 Use Cases

| **Domain** | **Application** | **Key Features** |
|------------|-----------------|------------------|
| 🤖 **Machine Learning** | Price prediction, exposure analysis | Rich feature vectors, contextual embeddings |
| 📊 **Research** | Propensity score matching | Urban covariates, accessibility metrics |
| 🏙️ **Urban Planning** | Accessibility research, zoning analysis | Spatial patterns, connectivity measures |
| 🧠 **AI/ML** | Neural networks, spatial clustering | Environmental context, amenity features |

## ✨ Why GeoFeatureKit?

| **Advantage** | **Benefit** |
|---------------|-------------|
| ✅ **Simple** | Just coordinates in – structured features out |
| ✅ **Powerful** | Dozens of geospatial metrics in one function call |
| ✅ **User-friendly** | Optional progress bars and verbose modes |
| ✅ **Open Data** | Built entirely on OSM and public geospatial libraries |

## 🚀 Quick Start

### Installation
```bash
pip install geofeaturekit
```

### Basic Usage

```python
from geofeaturekit import features_from_location

# Example: Analyze Times Square with progress bar
features = features_from_location({
    'latitude': 40.7580,
    'longitude': -73.9855,
    'radius_meters': 500
}, show_progress=True)

print(features)
```

## 📝 Example Output

**Times Square Analysis (500m radius):**
```json
{
  "network_metrics": {
    "basic_metrics": {
      "total_nodes": 777,
      "total_street_segments": 2313,
      "total_intersections": 0,
      "total_dead_ends": 41,
      "total_street_length_meters": 80044.7
    },
    "density_metrics": {
      "intersections_per_sqkm": 0.0,
      "street_length_per_sqkm": 101.916091
    },
    "connectivity_metrics": {
      "streets_to_nodes_ratio": 1.488417,
      "average_connections_per_node": {
        "value": 3.589,
        "confidence_interval_95": {
          "lower": 3.536,
          "upper": 3.643
        }
      }
    },
    "street_pattern_metrics": {
      "street_segment_length_distribution": {
        "minimum_meters": 0.5,
        "maximum_meters": 286.6,
        "mean_meters": 34.6,
        "median_meters": 12.0,
        "std_dev_meters": 50.7
      },
      "street_bearing_distribution": {
        "mean_degrees": 163.3,
        "std_dev_degrees": 101.5
      },
      "ninety_degree_intersection_ratio": 0.0,
      "bearing_entropy": 2.056
    }
  },
  "poi_metrics": {
    "absolute_counts": {
      "total_points_of_interest": 1076,
      "counts_by_category": {
        "total_restaurant_places": {
          "count": 173,
          "percentage": 16.1
        },
        "total_fast_food_places": {
          "count": 77,
          "percentage": 7.2
        },
        "total_cafe_places": {
          "count": 74,
          "percentage": 6.9
        },
        "total_bicycle_parking_places": {
          "count": 71,
          "percentage": 6.6
        },
        "total_bench_places": {
          "count": 27,
          "percentage": 2.5
        },
        "total_bar_places": {
          "count": 26,
          "percentage": 2.4
        },
        "total_bank_places": {
          "count": 24,
          "percentage": 2.2
        },
        "total_pub_places": {
          "count": 19,
          "percentage": 1.8
        },
        "total_bicycle_rental_places": {
          "count": 15,
          "percentage": 1.4
        },
        "total_theatre_places": {
          "count": 12,
          "percentage": 1.1
        },
        "total_pharmacy_places": {
          "count": 6,
          "percentage": 0.6
        },
        "total_atm_places": {
          "count": 4,
          "percentage": 0.4
        }
      }
    },
    "density_metrics": {
      "points_of_interest_per_sqkm": 1370.700637,
      "density_by_category": {
        "restaurant_places_per_sqkm": 220.382166,
        "fast_food_places_per_sqkm": 98.089172,
        "cafe_places_per_sqkm": 94.267516,
        "bicycle_parking_places_per_sqkm": 90.44586,
        "bank_places_per_sqkm": 30.573248,
        "theatre_places_per_sqkm": 15.286624,
        "pharmacy_places_per_sqkm": 7.643312
      }
    },
    "distribution_metrics": {
      "unique_category_count": 42,
      "largest_category": {
        "name": "restaurant",
        "count": 173,
        "percentage": 16.1
      },
      "diversity_metrics": {
        "shannon_diversity_index": 2.245,
        "simpson_diversity_index": 0.79,
        "category_evenness": 0.601
      },
      "spatial_distribution": {
        "mean_nearest_neighbor_distance_meters": 45.2,
        "nearest_neighbor_distance_std_meters": 28.7,
        "r_statistic": 0.68,
        "pattern_interpretation": "clustered"
      }
    }
  }
}
```

### 🔍 **Analysis Results**

| **Location Characteristics** | **Value** | **Interpretation** |
|---------------------------|-----------|-------------------|
| 🏙️ **POI Density** | 1,371 per km² | Ultra-dense location (rural areas: <10) |
| 🍽️ **Food Scene** | 324 establishments | Dining powerhouse in 500m radius |
| 🚲 **Transit Access** | 86 bike facilities | Sustainable transport infrastructure |
| 🏛️ **Entertainment** | 12 theaters + 38 venues | Major entertainment district |
| 🏪 **Financial Services** | 24 banks + 4 ATMs | Active commercial hub |

| **Network Intelligence** | **Value** | **Interpretation** |
|--------------------------|-----------|-------------------|
| 🚶 **Walkability** | 3.59 connections/node | High pedestrian connectivity |
| 🗺️ **Street Pattern** | 2.056 bearing entropy | Organized grid-like layout |
| 🛣️ **Network Density** | 101.9 km/km² | Dense street network |

| **Spatial Intelligence** | **Value** | **Use Case** |
|--------------------------|-----------|--------------|
| 📊 **Shannon Diversity** | 2.245 | High variety → Rich ML features |
| 📈 **Simpson Diversity** | 0.79 | Robust POI mix → Stable predictions |
| 🎯 **Clustering Pattern** | R = 0.68 | Distinct activity zones → Zoning analysis |

> **Perfect for:** Price prediction models, accessibility scoring, urban planning analysis

## 🎯 Key Features

### **Rich POI Analysis** *(Points of Interest)*
- **40+ categories**: restaurants, hospitals, schools, transit, entertainment
- **Density metrics**: POIs per square kilometer by category
- **Diversity indices**: 
  - *Shannon diversity*: Measures variety and evenness (higher = more diverse)
  - *Simpson diversity*: Probability two random POIs are different types
- **Spatial patterns**: clustered, dispersed, or random POI distributions

### **Street Network Insights**
- **Connectivity**: average connections per intersection
- **Total length**: meters of streets within radius
- **Segment patterns**: distribution of street segment lengths
- **Bearing analysis**: street orientation entropy and grid patterns

### **Progress Tracking**
| **Mode** | **Code** | **Use Case** |
|----------|----------|--------------|
| **Standard** | `show_progress=True, progress_detail='normal'` | General use with progress bars |
| **Verbose** | `show_progress=True, progress_detail='verbose'` | Detailed debugging information |
| **Silent** | `show_progress=False` | Batch processing, production |

```python
# Example: Verbose progress tracking
features = features_from_location(location, show_progress=True, progress_detail='verbose')
```

## 🔬 Scientific Applications

**Geospatial Research:**
```python
# Compare neighborhood walkability
locations = [
    {'latitude': 40.7580, 'longitude': -73.9855, 'radius_meters': 800},  # Times Square
    {'latitude': 40.7829, 'longitude': -73.9654, 'radius_meters': 800}   # Central Park
]

for loc in locations:
    features = features_from_location(loc)
    walkability_score = (
        features['poi_metrics']['density_metrics']['points_of_interest_per_sqkm'] * 0.4 +
        features['network_metrics']['connectivity_metrics']['average_connections_per_node']['value'] * 100 * 0.6
    )
    print(f"Walkability score: {walkability_score:.1f}")
```

**ML Feature Engineering:**
```python
# Generate features for price prediction model
import pandas as pd

properties = pd.read_csv('real_estate.csv')  # lat, lon, price columns
features_list = []

for _, row in properties.iterrows():
    location_features = features_from_location({
        'latitude': row['lat'],
        'longitude': row['lon'], 
        'radius_meters': 1000
    }, show_progress=False)
    
    # Extract key features for ML
    features_list.append({
        'restaurant_density': location_features['poi_metrics']['density_metrics']['restaurant_places_per_sqkm'],
        'transit_access': location_features['poi_metrics']['absolute_counts']['counts_by_category'].get('total_bus_station_places', {}).get('count', 0),
        'street_connectivity': location_features['network_metrics']['connectivity_metrics']['average_connections_per_node']['value'],
        'location_diversity': location_features['poi_metrics']['distribution_metrics']['diversity_metrics']['shannon_diversity_index']
    })

# Add to your ML pipeline
features_df = pd.DataFrame(features_list)
properties = pd.concat([properties, features_df], axis=1)
```

## 🛠 Advanced Usage

### **Batch Processing**
```python
# Process multiple locations efficiently
locations = [
    {'latitude': 40.7580, 'longitude': -73.9855, 'radius_meters': 500},
    {'latitude': 40.7829, 'longitude': -73.9654, 'radius_meters': 500},
    {'latitude': 40.7527, 'longitude': -73.9772, 'radius_meters': 500}
]

results = features_from_location(locations, show_progress=True)
```

### **Command Line Interface**
```bash
# Single location analysis
geofeaturekit analyze 40.7580 -73.9855 --radius 500 --verbose

# Batch analysis from file
geofeaturekit batch-analyze locations.json --radius 1000 --output results/
```

### **Custom Radius Analysis**
```python
# Compare different scales
radii = [200, 500, 1000, 2000]  # meters

for radius in radii:
    features = features_from_location({
        'latitude': 40.7580,
        'longitude': -73.9855, 
        'radius_meters': radius
    })
    
    poi_count = features['poi_metrics']['absolute_counts']['total_points_of_interest']
    print(f"{radius}m radius: {poi_count} POIs")
```

## 📖 Key Terms

| **Term** | **Definition** | **Scale** |
|----------|----------------|-----------|
| **POI** | Points of Interest (restaurants, hospitals, schools, ATMs) | Count |
| **Shannon Diversity** | Measures variety and evenness of POI types | 0-4+ (higher = more diverse) |
| **Simpson Diversity** | Probability two random POIs are different types | 0-1 (higher = more diverse) |
| **Bearing Entropy** | Street grid organization measure | 0-4+ (lower = more organized) |
| **R-statistic** | Spatial clustering pattern | 0-2.1 (<1 clustered, ~1 random, >1 dispersed) |
| **Connectivity** | Average connections per street intersection | 2-8+ (higher = more walkable) |

## 📊 Output Structure

GeoFeatureKit returns a comprehensive dictionary with four main sections:

```python
{
    'network_metrics': {
        'basic_metrics': {...},      # Node/edge counts, total length
        'density_metrics': {...},    # Per-km² measurements  
        'connectivity_metrics': {...}, # Connection patterns
        'street_pattern_metrics': {...} # Orientation, segment analysis
    },
    'poi_metrics': {
        'absolute_counts': {...},    # Raw POI counts by category
        'density_metrics': {...},    # POIs per km² by category
        'distribution_metrics': {...} # Diversity and spatial patterns
    },
    'units': {
        'area': 'square_meters',
        'length': 'meters', 
        'density': 'per_square_kilometer'
    }
}
```

## 🌍 Standards & Quality

- **SI Units**: All measurements in meters, square kilometers
- **Confidence Intervals**: Statistical uncertainty for network metrics
- **Reproducible**: Deterministic results with caching
- **Validated**: Comprehensive test suite with property-based testing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with [OSMnx](https://github.com/gboeing/osmnx), [NetworkX](https://github.com/networkx/networkx), and [GeoPandas](https://github.com/geopandas/geopandas). Data from [OpenStreetMap](https://www.openstreetmap.org/) contributors.

## 📚 Citation

If you use GeoFeatureKit in your research, please cite:

```bibtex
@software{geofeaturekit2025,
    title={GeoFeatureKit: Geospatial Feature Extraction and Analysis},
    author={Alexander Li},
    year={2025},
    url={https://github.com/lihangalex/geofeaturekit}
}
```

---

**Ready to analyze any location? Start with `pip install geofeaturekit` and explore geospatial patterns like never before! 🌍** 