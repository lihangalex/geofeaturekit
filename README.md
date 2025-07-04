# GeoFeatureKit

[![PyPI version](https://badge.fury.io/py/geofeaturekit.svg)](https://badge.fury.io/py/geofeaturekit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Python package for extracting and analyzing geospatial features from OpenStreetMap data.

## Features

- Extract street network metrics and points of interest around locations
- Analyze network centrality, density, and road distribution
- Generate network embeddings for machine learning
- Flexible output options: in-memory or disk storage
- Efficient handling of large datasets
- Simple, intuitive API

## Installation

You can install GeoFeatureKit using pip:

```bash
pip install geofeaturekit
```

Or install from source:

```bash
git clone https://github.com/alexanderli/geofeaturekit.git
cd geofeaturekit
pip install -e .
```

## Usage

### Python API

```python
from geofeaturekit import GeospatialFeatureExtractor
from geofeaturekit.core.config import AnalysisConfig

# Configure the analysis
config = AnalysisConfig(
    radius_meters=300,  # Analysis radius in meters
    output_dir="./results"  # Optional: Directory to save results
)

# Initialize the extractor
extractor = GeospatialFeatureExtractor(config=config)

# Analyze a single location
location = {
    "latitude": 43.6532,
    "longitude": -79.3832
}
results = extractor.extract_features(location)
```

### Command Line Interface

Analyze a single location and display results:
```bash
geofeaturekit --lat 43.6532 --lon -79.3832 --in-memory
```

Save results to default output directory:
```bash
geofeaturekit --lat 43.6532 --lon -79.3832
```

Customize analysis area and output location:
```bash
geofeaturekit --lat 43.6532 --lon -79.3832 --radius 1000 --save-dir ./my_results
```

Process multiple locations from a CSV file:
```bash
geofeaturekit --input locations.csv --save-dir ./results
```

## Results

GeoFeatureKit provides flexible options for handling results:

1. In-memory (default for API, opt-in for CLI):
   - Results are returned directly as Python objects
   - Best for small to medium datasets
   - Interactive analysis and immediate use

2. Disk storage (default for CLI, optional for API):
   - Results saved as JSON files
   - Best for large datasets
   - Each location saved in separate file
   - Efficient memory usage
   - Default directory: `output/`

Choose the approach that best fits your needs:
- Small dataset → Use in-memory
- Large dataset → Use disk storage
- Unsure? Start with default disk storage

## API Reference

### GeospatialFeatureExtractor

```python
def extract_features(
    locations: Union[Dict[str, float], List[Dict[str, float]]],
    config: Optional[AnalysisConfig] = None
) -> Optional[Union[AnalysisResults, List[AnalysisResults]]]
```

Parameters:
- `locations`: Single location dict or list of location dicts with 'latitude' and 'longitude'
- `config`: Optional AnalysisConfig object for customizing analysis parameters (default: None)

Returns:
- If `output_format` is 'memory': Single AnalysisResults object or list of AnalysisResults objects
- If `output_format` is 'json': None (results are saved to disk)

### AnalysisConfig

```python
def __init__(
    radius_meters: int = 300,  # Default analysis radius in meters
    enable_embeddings: bool = True,  # Enable network embeddings
    enable_urban_metrics: bool = True,  # Enable urban metrics computation
    output_format: str = 'memory'  # 'memory' or 'json'
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

The project includes several third-party packages with their own licenses - see the [NOTICE.md](NOTICE.md) file for details. 