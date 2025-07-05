"""Command line interface for GeoFeatureKit."""

import os
import json
import click
from typing import Tuple
from ..core.extractor import UrbanFeatureExtractor

@click.group()
def cli():
    """GeoFeatureKit - Urban feature extraction and analysis."""
    pass

@cli.command()
@click.argument('latitude', type=float)
@click.argument('longitude', type=float)
@click.option('--radius', '-r', default=1000,
              help='Analysis radius in meters (default: 1000)')
@click.option('--network-type', '-n', default='all',
              type=click.Choice(['drive', 'walk', 'bike', 'all']),
              help='Street network type (default: all)')
@click.option('--output', '-o', default='output',
              help='Output directory (default: output)')
@click.option('--cache/--no-cache', default=True,
              help='Use cache for downloaded data (default: True)')
def analyze(
    latitude: float,
    longitude: float,
    radius: int,
    network_type: str,
    output: str,
    cache: bool
):
    """Analyze urban features for a location.
    
    Example:
        geofeaturekit analyze 40.7580 -73.9855 --radius 500
    """
    # Create output directory
    os.makedirs(output, exist_ok=True)
    
    # Initialize extractor
    extractor = UrbanFeatureExtractor(
        cache_dir='cache' if cache else None,
        use_cache=cache
    )
    
    # Extract features
    click.echo(f"Analyzing location ({latitude}, {longitude})...")
    results = extractor.extract_features(
        location=(latitude, longitude),
        radius=radius,
        network_type=network_type
    )
    
    # Save results
    output_file = os.path.join(
        output,
        f"analysis_{latitude:.4f}_{longitude:.4f}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    click.echo(f"Results saved to {output_file}")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--radius', '-r', default=1000,
              help='Analysis radius in meters (default: 1000)')
@click.option('--network-type', '-n', default='all',
              type=click.Choice(['drive', 'walk', 'bike', 'all']),
              help='Street network type (default: all)')
@click.option('--output', '-o', default='output',
              help='Output directory (default: output)')
@click.option('--cache/--no-cache', default=True,
              help='Use cache for downloaded data (default: True)')
def batch_analyze(
    input_file: str,
    radius: int,
    network_type: str,
    output: str,
    cache: bool
):
    """Analyze multiple locations from a JSON file.
    
    The input file should contain a list of objects with 'latitude'
    and 'longitude' fields.
    
    Example:
        geofeaturekit batch-analyze locations.json --radius 500
    """
    # Create output directory
    os.makedirs(output, exist_ok=True)
    
    # Load locations
    with open(input_file, 'r') as f:
        locations = json.load(f)
    
    # Validate input format
    if not isinstance(locations, list):
        raise click.BadParameter("Input file must contain a JSON array")
    
    # Convert to tuples
    location_tuples = []
    for loc in locations:
        try:
            lat = float(loc['latitude'])
            lon = float(loc['longitude'])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError
            location_tuples.append((lat, lon))
        except (KeyError, ValueError):
            click.echo(f"Warning: Skipping invalid location {loc}")
    
    if not location_tuples:
        raise click.BadParameter("No valid locations found in input file")
    
    # Initialize extractor
    extractor = UrbanFeatureExtractor(
        cache_dir='cache' if cache else None,
        use_cache=cache
    )
    
    # Extract features
    click.echo(f"Analyzing {len(location_tuples)} locations...")
    results = extractor.batch_extract_features(
        locations=location_tuples,
        radius=radius,
        network_type=network_type
    )
    
    # Save results
    output_file = os.path.join(output, "batch_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    click.echo(f"Results saved to {output_file}")

if __name__ == '__main__':
    cli() 