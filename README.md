# GeoFeatureKit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/geofeaturekit.svg)](https://pypi.org/project/geofeaturekit/)
[![PyPI downloads](https://img.shields.io/pypi/dm/geofeaturekit.svg)](https://pypi.org/project/geofeaturekit/)
[![Tests](https://github.com/lihangalex/geofeaturekit/workflows/Test/badge.svg)](https://github.com/lihangalex/geofeaturekit/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GeoFeatureKit turns raw coordinates into rich, structured geospatial features – instantly.**

## 🎯 What You Get

**Input:** Just latitude and longitude coordinates  
**Output:** Comprehensive geospatial intelligence including:

- **🚀 NEW: Multi-modal isochrone accessibility**: Walk, bike, and drive accessibility analysis with custom speeds
- **17 Enhanced POI categories**: dining, retail, education, healthcare, culture, recreation, transportation, public transit, water features, green infrastructure, community, financial, accommodation, services, utilities, safety & emergency, and natural features
- **Street network metrics**: connectivity, total street length, segment distributions, pattern entropy
- **Spatial intelligence**: POI diversity indices (Shannon, Simpson) and clustering patterns

## 🚀 Use Cases

| **Domain** | **Application** | **Key Features** |
|------------|-----------------|------------------|
| 🤖 **Machine Learning** | Price prediction, exposure analysis | Rich feature vectors, contextual embeddings |
| 📊 **Research** | Propensity score matching | Urban covariates, accessibility metrics |
| 🏙️ **Urban Planning** | Accessibility research, zoning analysis | Spatial patterns, connectivity measures |
| 🧠 **AI/ML** | Neural networks, spatial clustering | Environmental context, amenity features |

## 🔧 Recent Updates

✅ **Fixed spatial distribution bug**: Mean nearest neighbor distance now correctly calculates distances to ALL other points, not just subsequent ones  
✅ **Improved coordinate detection**: Better handling of meter vs degree coordinate systems  
✅ **Enhanced precision**: Cleaner formatting with appropriate decimal places  
✅ **Fixed network metrics**: Corrected dead end and intersection counting logic  
✅ **Robust testing**: Replaced flaky tests with deterministic grid-based validation  
✅ **Python 3.9+ compatibility**: Full support across Python versions
✅ **Automated releases**: GitHub Actions now automatically publishes to PyPI on version tags

## 🌟 Enhanced Features

### 🏙️ **17 Comprehensive POI Categories**

| **Category** | **Examples** | **Use Cases** |
|-------------|--------------|---------------|
| 🍽️ **Dining** | Restaurants, cafes, bars, fast food | Food accessibility, nightlife analysis |
| 🏪 **Retail** | Supermarkets, malls, convenience stores | Shopping accessibility, commercial zones |
| 🎓 **Education** | Schools, universities, libraries | Educational accessibility, learning hubs |
| 🏥 **Healthcare** | Hospitals, clinics, pharmacies | Medical accessibility, health services |
| 🎭 **Culture** | Museums, theaters, art centers | Cultural richness, entertainment venues |
| 🏃 **Recreation** | Parks, gyms, sports centers | Fitness accessibility, recreational spaces |
| 🚗 **Transportation** | Parking, bike rental, airports | Mobility infrastructure, transport hubs |
| 🚇 **Public Transit** | Subway stations, bus stops, trams | Public transport accessibility |
| 🌊 **Water Features** | Rivers, fountains, coastlines | Natural water access, scenic features |
| 🌳 **Green Infrastructure** | Trees, parks, gardens, benches | Environmental quality, green spaces |
| 🏛️ **Community** | Community centers, places of worship | Social infrastructure, civic spaces |
| 🏦 **Financial** | Banks, ATMs, financial services | Banking accessibility, financial hubs |
| 🏨 **Accommodation** | Hotels, hostels, guest houses | Tourism infrastructure, lodging |
| 🔧 **Services** | Post offices, laundry, salons | Daily services, convenience access |
| ⚡ **Utilities** | Power stations, water treatment | Critical infrastructure, urban systems |
| 🚨 **Safety & Emergency** | Fire stations, police, hospitals | Emergency services, public safety |
| 🌿 **Natural** | Forests, beaches, nature reserves | Natural environment, biodiversity |

### 🚀 **Multi-Modal Isochrone Accessibility**

| **Mode** | **Default Speed** | **Use Cases** |
|----------|------------------|---------------|
| 🚶 **Walking** | 5.0 km/h | Pedestrian accessibility, walkability analysis |
| 🚴 **Biking** | 15.0 km/h | Cycling infrastructure, bike-friendly areas |
| 🚗 **Driving** | 40.0 km/h | Car accessibility, service area analysis |

- **Network-based routing**: Uses actual street networks for realistic travel times
- **Custom speed configuration**: Adjust speeds for different analysis scenarios
- **Combined analysis**: Compare radius-based vs time-based accessibility
- **Comprehensive metrics**: POI counts, area coverage, accessibility comparisons

### 🔬 **Advanced Spatial Analysis**

- **Nearest Neighbor Analysis**: Spatial clustering patterns (clustered/random/dispersed)
- **Diversity Metrics**: Shannon and Simpson indices for POI variety
- **Street Pattern Analysis**: Bearing entropy, intersection ratios, connectivity metrics
- **Graceful Error Handling**: Robust extraction even with limited data availability

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

### Enhanced POI Analysis

```python
from geofeaturekit import features_from_location

# Get comprehensive POI analysis
result = features_from_location({
    'latitude': 40.7580,  # Times Square
    'longitude': -73.9855,
    'radius_meters': 800
})

# Access enhanced POI categories
poi_counts = result['poi_metrics']['absolute_counts']['counts_by_category']

# Example: Find areas with good public transit
transit_count = poi_counts.get('total_public_transit_places', {}).get('count', 0)
print(f"Public transit stops: {transit_count}")

# Example: Analyze green infrastructure
green_count = poi_counts.get('total_green_infrastructure_places', {}).get('count', 0)
print(f"Green infrastructure: {green_count}")

# Example: Check water features
water_count = poi_counts.get('total_water_features_places', {}).get('count', 0)
print(f"Water features: {water_count}")

# Access spatial distribution analysis
spatial = result['poi_metrics']['distribution_metrics']['spatial_distribution']
print(f"Spatial pattern: {spatial['pattern_interpretation']}")
print(f"Mean distance between POIs: {spatial['mean_nearest_neighbor_distance_meters']}m")
```

### 🚀 **NEW: Multi-Modal Isochrone Accessibility Analysis**

```python
from geofeaturekit import features_from_coordinate

# Multi-modal accessibility analysis
features = features_from_coordinate(
    lat=40.7580,  # Times Square
    lon=-73.9855,
    max_travel_time_min_walk=10,    # 10-minute walking isochrone
    max_travel_time_min_bike=5,     # 5-minute biking isochrone  
    max_travel_time_min_drive=15,   # 15-minute driving isochrone
    speed_config={'walk': 4.8, 'bike': 17, 'drive': 35}  # Custom speeds
)

# Access walking accessibility
walk_data = features['isochrone_features_walk']
walk_pois = walk_data['poi_metrics']['absolute_counts']['total_points_of_interest']
walk_area = walk_data['isochrone_info']['area_sqm']
print(f"Walking (10min): {walk_pois} POIs accessible in {walk_area:.0f} sqm")

# Access biking accessibility  
bike_data = features['isochrone_features_bike']
bike_pois = bike_data['poi_metrics']['absolute_counts']['total_points_of_interest']
bike_area = bike_data['isochrone_info']['area_sqm']
print(f"Biking (5min): {bike_pois} POIs accessible in {bike_area:.0f} sqm")

# Compare accessibility by transportation mode
print("Accessibility Comparison:")
for mode, data in features.items():
    info = data['isochrone_info']
    poi_count = data['poi_metrics']['absolute_counts']['total_points_of_interest']
    print(f"  {info['mode'].title()}: {poi_count} POIs in {info['travel_time_minutes']}min")
```

### Combined Radius + Isochrone Analysis

```python
from geofeaturekit import features_from_coordinate

# Analyze both circular radius and accessibility isochrones
features = features_from_coordinate(
    lat=40.7580,
    lon=-73.9855,
    radius_m=500,                   # 500m circular radius
    max_travel_time_min_walk=8,     # 8-minute walking accessibility
    max_travel_time_min_bike=4      # 4-minute biking accessibility
)

# Compare different analysis methods
radius_pois = features['radius_features']['poi_metrics']['absolute_counts']['total_points_of_interest']
walk_pois = features['isochrone_features_walk']['poi_metrics']['absolute_counts']['total_points_of_interest'] 
bike_pois = features['isochrone_features_bike']['poi_metrics']['absolute_counts']['total_points_of_interest']

print(f"Circular (500m): {radius_pois} POIs")
print(f"Walking (8min): {walk_pois} POIs") 
print(f"Biking (4min): {bike_pois} POIs")
```

## 📝 Example Output

**Times Square Analysis (500m radius):**
```json
{
  "network_metrics": {
    "basic_metrics": {
      "total_nodes": 777,
      "total_street_segments": 2313,
      "total_intersections": 731,
      "total_dead_ends": 0,
      "total_street_length_meters": 80044.7
    },
    "density_metrics": {
      "intersections_per_sqkm": 930.74,
      "street_length_per_sqkm": 101.92
    },
    "connectivity_metrics": {
              "streets_to_nodes_ratio": 1.488,
        "average_connections_per_node": {
          "value": 5.954,
          "confidence_interval_95": {
            "lower": 5.837,
            "upper": 6.071
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
        "total_dining_places": {
          "count": 370,
          "percentage": 34.39
        },
        "total_transportation_places": {
          "count": 190,
          "percentage": 17.66
        },
        "total_public_transit_places": {
          "count": 96,
          "percentage": 8.92
        },
        "total_green_infrastructure_places": {
          "count": 37,
          "percentage": 3.44
        },
        "total_culture_places": {
          "count": 34,
          "percentage": 3.16
        },
        "total_financial_places": {
          "count": 31,
          "percentage": 2.88
        },
        "total_retail_places": {
          "count": 18,
          "percentage": 1.67
        },
        "total_accommodation_places": {
          "count": 16,
          "percentage": 1.49
        },
        "total_services_places": {
          "count": 16,
          "percentage": 1.49
        },
        "total_healthcare_places": {
          "count": 13,
          "percentage": 1.21
        },
        "total_water_features_places": {
          "count": 11,
          "percentage": 1.02
        },
        "total_recreation_places": {
          "count": 6,
          "percentage": 0.56
        },
        "total_education_places": {
          "count": 3,
          "percentage": 0.28
        },
        "total_community_places": {
          "count": 3,
          "percentage": 0.28
        }
      }
    },
    "density_metrics": {
      "points_of_interest_per_sqkm": 1370.7,
      "density_by_category": {
        "dining_places_per_sqkm": 471.1,
        "transportation_places_per_sqkm": 241.9,
        "public_transit_places_per_sqkm": 122.2,
        "green_infrastructure_places_per_sqkm": 47.1,
        "culture_places_per_sqkm": 43.3,
        "financial_places_per_sqkm": 39.5,
        "retail_places_per_sqkm": 22.9,
        "water_features_places_per_sqkm": 14.0
      }
    },
    "distribution_metrics": {
      "unique_category_count": 15,
      "largest_category": {
        "name": "dining",
        "count": 370,
        "percentage": 34.39
      },
      "diversity_metrics": {
        "shannon_diversity_index": 2.11,
        "simpson_diversity_index": 0.81,
        "category_evenness": 0.78
      },
      "spatial_distribution": {
        "mean_nearest_neighbor_distance_meters": 13.2,
        "nearest_neighbor_distance_std_meters": 9.7,
        "r_statistic": 0.978,
        "pattern_interpretation": "random"
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
| 🚶 **Walkability** | 5.95 connections/node | Very high pedestrian connectivity |
| 🗺️ **Street Pattern** | 2.056 bearing entropy | Organized grid-like layout |
| 🛣️ **Network Density** | 101.9 km/km² | Dense street network |

| **Spatial Intelligence** | **Value** | **Use Case** |
|--------------------------|-----------|--------------|
| 📊 **Shannon Diversity** | 2.245 | High variety → Rich ML features |
| 📈 **Simpson Diversity** | 0.79 | Robust POI mix → Stable predictions |
| 🎯 **Clustering Pattern** | R = 0.978 | Random distribution → Uniform coverage |

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

## 🚀 Automated Releases

GeoFeatureKit uses automated releases via GitHub Actions. Every time a version tag is pushed, the package is automatically:

- ✅ **Tested** on Python 3.9, 3.10, 3.11, and 3.12  
- ✅ **Built** with proper validation  
- ✅ **Published** to PyPI  
- ✅ **Released** on GitHub with auto-generated notes  

For maintainers: Use `./release.sh <version>` to automate the entire release process.

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