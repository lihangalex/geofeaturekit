# GeoFeatureKit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/geofeaturekit.svg)](https://pypi.org/project/geofeaturekit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/lihangalex/geofeaturekit)

**GeoFeatureKit transforms simple coordinates into powerful urban insights. Analyze street networks, POI diversity, and spatial patterns effortlessly – no paid APIs or complex setup required.**

✅ **Simple:** Just coordinates in – rich features out  
✅ **Powerful:** Advanced urban metrics in one function call  
✅ **Open Data:** Built entirely on OSM and public geospatial libraries

---

## 🚀 Quick Start

Install GeoFeatureKit:

```bash
pip install geofeaturekit
```

Extract features for any location:

```python
from geofeaturekit import features_from_location

# Example: Analyze Central Park
features = features_from_location({
    'latitude': 40.7829,
    'longitude': -73.9654,
    'radius_meters': 500
})

network = features['network_metrics']
pois = features['poi_metrics']

# Print key results
print(f"Total street length: {network['basic_metrics']['total_street_length_meters']:.1f} m")
print(f"Average connections per node: {network['connectivity_metrics']['average_connections_per_node']['value']:.2f} "
      f"[95% CI: {network['connectivity_metrics']['average_connections_per_node']['confidence_interval_95']['lower']:.2f} – "
      f"{network['connectivity_metrics']['average_connections_per_node']['confidence_interval_95']['upper']:.2f}]")

print(f"Total POIs: {pois['absolute_counts']['total_points_of_interest']}")
print("Top POI categories:")
for category, data in list(pois['absolute_counts']['counts_by_category'].items())[:3]:
    print(f"  • {category.replace('total_', '').replace('_places', '').replace('_', ' ').title()}: {data['count']} ({data['percentage']}%)")

print(f"Shannon diversity index: {pois['distribution_metrics']['diversity_metrics']['shannon_diversity_index']:.2f}")
print(f"Spatial pattern: {pois['distribution_metrics']['spatial_distribution']['pattern_interpretation']}")
```

**Example output:**
```
Total street length: 41,254.3 m
Average connections per node: 3.26 [95% CI: 3.17 – 3.34]

Total POIs: 185
Top POI categories:
  • Bench: 66 (35.7%)
  • Unknown: 52 (28.1%)
  • Waste Basket: 16 (8.6%)

Shannon diversity index: 1.95
Spatial pattern: clustered
```

## 📦 Installation

```bash
# Install from PyPI
pip install geofeaturekit

# Or install from GitHub for latest development version
pip install git+https://github.com/lihangalex/geofeaturekit.git

# For development
git clone https://github.com/lihangalex/geofeaturekit.git
cd geofeaturekit
pip install -e .
```

**Requirements:** Python 3.9+, NumPy, SciPy, GeoPandas, OSMnx, NetworkX

## 🔍 Key Features

### 🏙️ **Street Network Analysis**
- **Connectivity Metrics**: Streets-to-nodes ratios, average connections per node with confidence intervals
- **Pattern Analysis**: Street bearing distributions, entropy measures, grid pattern detection
- **Density Calculations**: Street length per km², intersection density, segment distributions
- **Statistical Rigor**: Confidence intervals, standard deviations, robust statistical measures

### 📍 **Points of Interest (POI) Analysis** 
- **Comprehensive Categorization**: 40+ POI categories with automatic classification
- **Density Metrics**: POI counts per km² with category-specific breakdowns
- **Diversity Analysis**: Shannon diversity index, Simpson diversity, category evenness
- **Spatial Distribution**: Nearest neighbor analysis, clustering patterns

### 📊 **Advanced Urban Metrics**
- **Data Quality Assessment**: Completeness percentages, reliability scores
- **Statistical Analysis**: Confidence intervals for estimated metrics only
- **Spatial Analysis**: Area calculations, density distributions, pattern recognition
- **Real-world Validation**: Tested on major urban areas worldwide

## 🌟 Real-World Examples

### Times Square Analysis
```python
# Dense commercial district
features = features_from_location({
    'latitude': 40.7580, 'longitude': -73.9855, 'radius_meters': 500
})

# Results:
# - 777 network nodes, 2,313 street segments
# - 80.0 km of streets in 0.785 km² area
# - 1,076 POIs (1,371 per km²)
# - 42 unique POI categories
# - High connectivity: 3.59 connections per node
```

### Central Park Analysis  
```python
# Park and recreational area
features = features_from_location({
    'latitude': 40.7829, 'longitude': -73.9654, 'radius_meters': 500  
})

# Results:
# - 356 network nodes, 1,002 street segments
# - 41.3 km of paths and streets
# - 185 POIs (236 per km²) 
# - Dominated by benches (35.7%) and recreational amenities
# - Lower but adequate connectivity: 3.26 connections per node
```

### Grand Central District
```python
# Transportation and business hub
features = features_from_location({
    'latitude': 40.7527, 'longitude': -73.9772, 'radius_meters': 500
})

# Results:
# - 1,002 network nodes, 2,975 street segments  
# - 91.2 km of streets (highest density)
# - 1,131 POIs (1,441 per km²)
# - Mixed commercial and transportation amenities
# - Excellent connectivity: 3.60 connections per node
```

## 📈 Output Structure

```python
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
                }
                // ... 40+ categories
            }
        },
        "density_metrics": {
            "points_of_interest_per_sqkm": 1370.700637,
            "density_by_category": {
                "restaurant_places_per_sqkm": 220.382166,
                "cafe_places_per_sqkm": 94.267516
                // ... per-category densities
            }
        },
        "distribution_metrics": {
            "unique_category_count": 42,
            "diversity_metrics": {
                "shannon_diversity_index": 2.245,
                "simpson_diversity_index": 0.79,
                "category_evenness": 0.601
            },
            "spatial_distribution": {
                "pattern_interpretation": "clustered"
            }
        }
    },
    "units": {
        "area": "square_meters",
        "length": "meters", 
        "density": "per_square_kilometer"
    }
}
```

## 🔬 Scientific Applications

### Urban Planning Research
```python
# Compare neighborhood walkability
locations = [
    {'name': 'Downtown', 'lat': 40.7580, 'lon': -73.9855},
    {'name': 'Residential', 'lat': 40.7829, 'lon': -73.9654}
]

for loc in locations:
    features = features_from_location(loc)
    connectivity = features['network_metrics']['connectivity_metrics']
    poi_density = features['poi_metrics']['density_metrics']
    
    print(f"{loc['name']} Walkability Score:")
    print(f"  Connectivity: {connectivity['average_connections_per_node']['value']:.2f}")
    print(f"  POI Density: {poi_density['points_of_interest_per_sqkm']:.0f} per km²")
```

### Accessibility Analysis
```python
# Analyze service accessibility
features = features_from_location({'lat': 40.7527, 'lon': -73.9772, 'radius_meters': 800})

essential_services = [
    'restaurant_places_per_sqkm',
    'bank_places_per_sqkm', 
    'pharmacy_places_per_sqkm'
]

for service in essential_services:
    density = features['poi_metrics']['density_metrics'][service]
    print(f"{service}: {density:.1f} per km²")
```

### Comparative Urban Studies
```python
# Multi-city comparison
cities = [
    {'name': 'NYC Times Square', 'lat': 40.7580, 'lon': -73.9855},
    {'name': 'London Piccadilly', 'lat': 51.5100, 'lon': -0.1347},
    {'name': 'Tokyo Shibuya', 'lat': 35.6598, 'lon': 139.7006}
]

results = {}
for city in cities:
    features = features_from_location(city)
    results[city['name']] = {
        'street_density': features['network_metrics']['density_metrics']['street_length_per_sqkm'],
        'poi_diversity': features['poi_metrics']['distribution_metrics']['diversity_metrics']['shannon_diversity_index']
    }
```

## 🛠️ Advanced Usage

### Batch Processing
```python
import pandas as pd

# Process multiple locations
locations_df = pd.read_csv('study_locations.csv')
results = []

for _, row in locations_df.iterrows():
    try:
        features = features_from_location({
            'latitude': row['lat'],
            'longitude': row['lon'], 
            'radius_meters': row['radius']
        })
        
        results.append({
            'location_id': row['id'],
            'poi_count': features['poi_metrics']['absolute_counts']['total_points_of_interest'],
            'street_length': features['network_metrics']['basic_metrics']['total_street_length_meters'],
            'connectivity': features['network_metrics']['connectivity_metrics']['average_connections_per_node']['value']
        })
    except Exception as e:
        print(f"Error processing {row['id']}: {e}")

results_df = pd.DataFrame(results)
```

### Statistical Analysis
```python
# Extract confidence intervals and statistical measures
features = features_from_location({'lat': 40.7580, 'lon': -73.9855, 'radius_meters': 500})

# Network connectivity with confidence intervals
conn = features['network_metrics']['connectivity_metrics']['average_connections_per_node']
print(f"Average connections: {conn['value']:.3f}")
print(f"95% CI: [{conn['confidence_interval_95']['lower']:.3f}, {conn['confidence_interval_95']['upper']:.3f}]")

# POI category analysis with exact counts
categories = features['poi_metrics']['absolute_counts']['counts_by_category']
for category, data in categories.items():
    print(f"{category}: {data['count']} ({data['percentage']:.1f}%)")
```

## 📋 Standards & Quality

**Metric Standards:** All metrics follow SI (International System of Units) standards:
- **Length**: meters (m) • **Area**: square meters (m²) • **Density**: per square kilometer (per km²)
- **Angles**: degrees (°) • **Statistical measures**: Include confidence intervals where statistically appropriate

**Testing & Quality:**
- Comprehensive test suite with property-based testing
- Real-world validation on major urban areas
- Statistical rigor with confidence intervals for estimated metrics only
- Robust error handling and performance optimization

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`tox -e py310`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenStreetMap**: For providing the foundational geographic data
- **OSMnx**: For excellent OpenStreetMap network analysis tools
- **GeoPandas**: For robust geospatial data processing
- **SciPy ecosystem**: For statistical analysis capabilities

## 📚 Citation

If you use GeoFeatureKit in your research, please cite:

```bibtex
@software{geofeaturekit2024,
    title={GeoFeatureKit: Urban Feature Extraction and Analysis},
    author={Your Name},
    year={2024},
    url={https://github.com/lihangalex/geofeaturekit}
}
```

---

**Ready to analyze your city? Start with `pip install geofeaturekit` and explore urban patterns like never before! 🏙️** 