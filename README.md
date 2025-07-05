# GeoFeatureKit

[![PyPI version](https://badge.fury.io/py/geofeaturekit.svg)](https://badge.fury.io/py/geofeaturekit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Python library for extracting and analyzing urban features from OpenStreetMap data. GeoFeatureKit helps urban planners, researchers, and developers analyze city infrastructure, amenities, and spatial patterns using standardized metrics.

## Features

- **Street Network Analysis**: Extract and analyze street network characteristics including:
  - Street lengths and connectivity
  - Network topology and patterns
  - Intersection analysis
  - Street network density metrics
  - Dead-end detection
  - Street bearing analysis

- **Points of Interest**: Analyze the distribution and density of amenities, services, and other POIs:
  - Automatic categorization by type
  - Density analysis by category
  - Accessibility metrics
  - Diversity measures
  - Custom category grouping
  - Nearest neighbor analysis

- **Land Use Analysis**:
  - Land use mix calculations
  - Area measurements
  - Boundary analysis
  - Usage percentages
  - Mixed-use detection

- **Data Quality Metrics**:
  - Completeness scores
  - Data reliability measures
  - Coverage analysis
  - Source verification

## Installation

```bash
# From PyPI
pip install geofeaturekit

# Or clone the repository for development
git clone https://github.com/yourusername/geofeaturekit.git
cd geofeaturekit
pip install -e .
```

## Quick Start

```python
from geofeaturekit import features_from_location

# Extract features for Times Square, New York
features = features_from_location(
    latitude=40.758,
    longitude=-73.9855,
    radius_meters=500,  # 500m radius around the point
    network_type='all'  # Include all street types
)

# Access specific metrics
network_metrics = features['metrics']['network_metrics']
poi_metrics = features['metrics']['poi_metrics']
land_use = features['metrics']['land_use_metrics']
```

## Example Analysis: Times Square

Let's analyze Times Square, one of New York City's most iconic locations:

```python
from geofeaturekit import features_from_location

# Analyze Times Square
features = features_from_location(
    latitude=40.758,
    longitude=-73.9855,
    radius_meters=500,  # 500m radius
    network_type='all'  # Include all street types
)

# The results include comprehensive metrics:
{
    "metadata": {
        "location": {
            "latitude": 40.758,
            "longitude": -73.9855
        },
        "radius_meters": 500,
        "network_type": "all",
        "area_sqm": 785398.2  # π * radius² = area of analysis
    },
    "metrics": {
        "network_metrics": {
            "basic_metrics": {
                "total_street_length_meters": 80044.7,  # Dense street network
                "total_intersections": 731,
                "total_dead_ends": 0,  # Well-connected streets
                "total_nodes": 777,
                "total_street_segments": 2313
            },
            "density_metrics": {
                "intersections_per_sqm": 0.000931,  # High intersection density
                "street_length_per_sqm": 0.101916,
                "nodes_per_sqm": 0.000989,
                "units": "per_square_meter"
            },
            "connectivity_metrics": {
                "average_connections_per_node": 3.411,  # Good connectivity
                "number_of_disconnected_networks": 1,  # Single connected network
                "number_of_street_loops": 1537,
                "streets_to_nodes_ratio": 2.977
            }
        },
        "poi_metrics": {
            "absolute_counts": {
                "total_points_of_interest": 1076,  # Very high POI count
                "counts_by_category": {
                    "total_restaurant_places": 173,
                    "total_fast_food_places": 77,
                    "total_cafe_places": 74,
                    "total_bicycle_parking_places": 71,
                    "total_bank_places": 24,
                    # ... more categories
                }
            },
            "density_metrics": {
                "points_of_interest_per_sqm": 0.00137,  # High POI density
                "density_by_category": {
                    "restaurant_places_per_sqm": 0.00022,
                    "fast_food_places_per_sqm": 0.000098,
                    "cafe_places_per_sqm": 0.000094
                }
            },
            "distribution_metrics": {
                "unique_category_count": 42,  # Diverse amenities
                "largest_category_count": 439,
                "largest_category_name": "unknown",
                "largest_category_count_percent": 40.8
            }
        },
        "pedestrian_network": {
            "intersection_spacing_meters": {
                "minimum": 0.8,
                "maximum": 286.6,
                "median": 12.1,  # Short blocks, walkable
                "standard_deviation": 51.5
            },
            "dead_end_street_count": 0,
            "total_dead_end_street_length_meters": 0.0
        },
        "land_use_metrics": {
            "area_measurements": {
                "total_area_sqm": 56.6,
                "commercial_area_sqm": 53.1,  # Predominantly commercial
                "open_space_area_sqm": 3.6
            }
        },
        "data_quality_metrics": {
            "data_completeness_percentages": {
                "percent_network_data_complete": 100.0,
                "percent_poi_data_complete": 100.0,
                "percent_land_use_data_complete": 100.0
            }
        }
    }
}
```

This example demonstrates several key features of Times Square's urban environment:

1. **Dense Street Network**:
   - 80km total street length in just 0.785 km² (500m radius)
   - 731 intersections with no dead ends
   - High connectivity (3.4 connections per node)

2. **Rich Point of Interest Density**:
   - 1,076 total POIs (1.37 POIs per 1000 m²)
   - 42 unique categories of amenities
   - High concentration of restaurants (173), cafes (74), and services

3. **Pedestrian-Friendly Design**:
   - Median intersection spacing of 12.1m
   - No dead-end streets
   - Dense network of pedestrian paths

4. **Commercial District**:
   - 93.8% commercial area (53.1/56.6 m²)
   - High concentration of retail and services
   - Mixed-use urban environment

5. **Data Quality**:
   - 100% complete network data
   - High reliability scores
   - Comprehensive POI coverage

This analysis reveals Times Square as a dense, highly connected commercial district with exceptional pedestrian accessibility and a diverse mix of amenities.

## Example Output

Here's a sample of what you'll get (truncated for readability):

```python
{
    "metadata": {
        "location": {
            "latitude": 40.758,
            "longitude": -73.9855
        },
        "radius_meters": 500,
        "network_type": "all",
        "area_sqm": 785398.2  # π * radius²
    },
    "metrics": {
        "network_metrics": {
            "basic_metrics": {
                "total_street_length_meters": 80044.7,
                "total_intersections": 731,
                "total_dead_ends": 0,
                "total_nodes": 777
            },
            "density_metrics": {
                "intersections_per_sqm": 0.000931,
                "street_length_per_sqm": 0.101916,
                "nodes_per_sqm": 0.000989,
                "units": "per_square_meter"
            }
        },
        "poi_metrics": {
            "absolute_counts": {
                "total_points_of_interest": 1076,
                "counts_by_category": {
                    "restaurant_places": 173,
                    "cafe_places": 74,
                    "bank_places": 24
                    # ... more categories
                }
            },
            "density_metrics": {
                "points_of_interest_per_sqm": 0.00137,
                "density_by_category": {
                    "restaurant_places_per_sqm": 0.00022,
                    "cafe_places_per_sqm": 0.000094
                    # ... more categories
                }
            }
        }
    }
}
```

## Common Use Cases

### 1. Neighborhood Comparison
```python
from geofeaturekit import features_from_location

# Compare two neighborhoods
locations = [
    (40.7829, -73.9654, "Central Park"),  # Central Park
    (40.7527, -73.9772, "Grand Central")  # Grand Central
]

comparisons = {}
for lat, lon, name in locations:
    features = features_from_location(
        latitude=lat,
        longitude=lon,
        radius_meters=500
    )
    comparisons[name] = features['metrics']

# Access specific metrics for comparison
for name, metrics in comparisons.items():
    poi_density = metrics['poi_metrics']['density_metrics']['points_of_interest_per_sqm']
    print(f"{name} POI Density: {poi_density:.6f} per m²")
```

### 2. Custom Analysis Area
```python
from geofeaturekit import features_from_polygon
import geopandas as gpd

# Load a custom polygon (e.g., neighborhood boundary)
area = gpd.read_file("my_neighborhood.geojson")
features = features_from_polygon(
    area.geometry[0],
    network_type='drive'  # Focus on driveable streets
)
```

### 3. Batch Processing
```python
from geofeaturekit import features_from_location
import pandas as pd

# Process multiple locations
locations_df = pd.read_csv("locations.csv")
results = []

for _, row in locations_df.iterrows():
    features = features_from_location(
        latitude=row['lat'],
        longitude=row['lon'],
        radius_meters=row['radius']
    )
    results.append({
        'location_id': row['id'],
        'features': features
    })
```

## Output Metrics

All metrics follow SI (International System of Units) standards.

### Network Metrics

Basic metrics:
- Total street length (meters)
- Number of intersections
- Number of nodes and edges
- Network connectivity measures

Density metrics (all per square meter):
- Street density (meters per m²)
- Intersection density (per m²)
- Node density (per m²)

Advanced metrics:
- Street network patterns
- Urban form characteristics
- Accessibility measures
- Street bearing distribution
- Dead-end analysis
- Network connectivity scores

### POI Metrics

Basic metrics:
- Total POI count
- Category distribution
- Unique amenity types
- Category hierarchies

Density measures (all per square meter):
- Total POI density (per m²)
- Category-specific densities (per m²)
- Service concentration metrics

Diversity metrics:
- Category mix
- Service variety
- Land use mix
- Entropy measures
- Specialization indices

### Area Measurements

All area measurements are in square meters (m²), following SI standards. This includes:
- Total area coverage
- Land use areas
- Service catchment areas
- Accessibility zones
- Building footprints
- Open space measurements

### Data Quality Indicators

The library provides quality metrics for:
- Data completeness (0-100%)
- Source reliability scores
- Temporal accuracy
- Spatial precision
- Coverage consistency

## Configuration

You can customize the analysis using configuration parameters:

```python
from geofeaturekit import set_config

set_config({
    'poi_categories': ['restaurant', 'cafe', 'retail'],  # Focus on specific POIs
    'network_types': ['drive', 'walk'],  # Street network types to include
    'min_intersection_spacing': 10,  # Minimum meters between intersections
    'custom_projections': True,  # Use custom projections for accurate areas
})
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The project includes several third-party packages with their own licenses - see the [NOTICE.md](NOTICE.md) file for details. 