# GeoFeatureKit

[![PyPI version](https://badge.fury.io/py/geofeaturekit.svg)](https://badge.fury.io/py/geofeaturekit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Python library for extracting and analyzing urban features from OpenStreetMap data.

## Features

- **Street Network Analysis**: Extract and analyze street network characteristics including:
  - Street lengths and connectivity
  - Network topology and patterns
  - Intersection analysis
  - Street network density metrics

- **Points of Interest**: Analyze the distribution and density of amenities, services, and other POIs
  - Categorization by type
  - Density analysis
  - Accessibility metrics
  - Diversity measures

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geofeaturekit.git
cd geofeaturekit

# Install in development mode
pip install -e .
```

## Quick Start

```python
from geofeaturekit import features_from_location

# Extract features for a location
features = features_from_location(
    latitude=40.7128,
    longitude=-74.0060,
    radius_meters=1000
)

# Access the results
network = features['network']
pois = features['pois']
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

### POI Metrics

Basic metrics:
- Total POI count
- Category distribution
- Unique amenity types

Density measures (all per square meter):
- Total POI density (per m²)
- Category-specific densities (per m²)

Diversity metrics:
- Category mix
- Service variety
- Land use mix

### Area Measurements

All area measurements are in square meters (m²), following SI standards. This includes:
- Total area coverage
- Land use areas
- Service catchment areas
- Accessibility zones

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The project includes several third-party packages with their own licenses - see the [NOTICE.md](NOTICE.md) file for details. 