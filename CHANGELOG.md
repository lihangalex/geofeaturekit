# Changelog

All notable changes to GeoFeatureKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-01-03

### Added
- **6 Critical New POI Categories**:
  - 🚻 toilets_hygiene: Public toilets, showers, chemists - ESSENTIAL amenities
  - 👶 childcare: Nurseries, kindergartens, daycare - Family support infrastructure  
  - 🚗 automotive: Gas stations, EV charging, car repair - Vehicle services
  - 🐕 animal_services: Veterinarians, pet shops, shelters - Pet care
  - 💼 workspace: Coworking spaces, offices - Modern work infrastructure
  - 🚴 bicycle_services: Bike shops, repairs, rentals - Cycling support

### Improved
- **Massive POI Coverage Improvements**:
  - Dining places: +118% better coverage (34→74 in Times Square)
  - Retail places: +2,600% better coverage (1→27 in Times Square)  
  - Transportation: +225% better coverage (4→13 in Times Square)
  - Overall categorization: 64%→87% success rate (+36% improvement)
  - Unknown places reduced: 35.8%→18.7% (-48% reduction)

### Enhanced
- **Expanded POI Tags**:
  - Enhanced retail tags: clothes, shoes, electronics, books, gifts
  - Expanded dining tags: bakery, deli, ice cream, food courts
  - Comprehensive services: beauty, massage, optician, travel agents
  - Better transportation coverage: bicycle shops, car services
- **Total POI Categories**: Expanded from 17→23 comprehensive categories

## [0.4.0] - 2025-01-02

### Added
- **🚀 Multi-modal isochrone accessibility analysis**:
  - Walk, bike, and drive accessibility with custom speeds
  - Network-based routing using actual street networks via OSMnx
  - Combined radius + isochrone analysis capability
  - Comprehensive speed configuration and validation

### Features
- **New `features_from_coordinate()` function**:
  - Support for `max_travel_time_min_walk/bike/drive` parameters
  - Default speeds: walk 5.0 km/h, bike 15.0 km/h, drive 40.0 km/h
  - Speed validation and error handling
  - Returns structured data with `isochrone_info` and full metrics

### Technical
- **Enhanced isochrone utility module** (`geofeaturekit/utils/isochrone.py`):
  - `calculate_isochrone_distance()` for time/speed to distance conversion
  - `create_isochrone_polygon()` for network-based routing
  - `extract_isochrone_features()` for comprehensive feature extraction
  - Fallback mechanisms for network routing failures

## [0.3.0] - 2025-01-01

### Fixed
- **Spatial distribution bug**: Mean nearest neighbor distance now correctly calculates distances to ALL other points, not just subsequent ones
- **Network metrics**: Corrected dead end and intersection counting logic
- **Coordinate detection**: Better handling of meter vs degree coordinate systems

### Improved
- **Enhanced precision**: Cleaner formatting with appropriate decimal places
- **Robust testing**: Replaced flaky tests with deterministic grid-based validation
- **Python 3.9+ compatibility**: Full support across Python versions

### Added
- **Automated releases**: GitHub Actions now automatically publishes to PyPI on version tags

## [0.2.4] - 2024-12-30

### Fixed
- **Flaky POI test**: Fixed spatial pattern interpretation test causing Python 3.10 GitHub Actions failures
- **Area calculation precision**: Improved accuracy for very small radii

### Documentation
- **Updated README**: Added Recent Updates section highlighting bug fixes
- **Enhanced examples**: Better documentation of key features

### Release
- **GitHub release v0.2.4**: Created with git tag and automated PyPI publishing
- **PyPI publication**: Successfully published updated package with documentation

## [0.2.0] - 2024-12-25

### Added
- **Comprehensive POI analysis**: 17 enhanced categories with density metrics
- **Spatial intelligence**: POI diversity indices (Shannon, Simpson) and clustering patterns
- **Street network insights**: Connectivity, bearing analysis, intersection ratios
- **Progress tracking**: Optional progress bars with normal/verbose modes

### Enhanced
- **Geospatial features**: Points of Interest, street networks, spatial patterns
- **Statistical analysis**: Confidence intervals, diversity measurements
- **User experience**: Comprehensive progress tracking and error handling

## [0.1.0] - 2024-12-20

### Added
- **Initial release**: Basic geospatial feature extraction from coordinates
- **Core functionality**: POI extraction, network analysis, spatial metrics
- **Open source**: MIT license, GitHub repository, PyPI package 