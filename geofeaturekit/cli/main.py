"""Command line interface for GeoFeatureKit."""

import os
import json
import click
from typing import Tuple
from ..core.extractor import UrbanFeatureExtractor

class FloatRange(click.ParamType):
    """Custom parameter type for float ranges."""
    
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val
        
    def convert(self, value, param, ctx):
        try:
            rv = float(value)
        except ValueError:
            self.fail(f'{value!r} is not a valid float', param, ctx)
        
        if self.min_val is not None and rv < self.min_val:
            self.fail(f'{value!r} is smaller than {self.min_val}', param, ctx)
        if self.max_val is not None and rv > self.max_val:
            self.fail(f'{value!r} is larger than {self.max_val}', param, ctx)
        
        return rv

# Custom types for latitude and longitude
LATITUDE = FloatRange(-90, 90)
LONGITUDE = FloatRange(-180, 180)

@click.group()
def cli():
    """GeoFeatureKit - Urban feature extraction and analysis."""
    pass

@cli.command()
@click.argument('latitude', type=LATITUDE)
@click.argument('longitude', type=LONGITUDE)
@click.option('--radius', '-r', default=1000,
              help='Analysis radius in meters (default: 1000)')
@click.option('--network-type', '-n', default='all',
              type=click.Choice(['drive', 'walk', 'bike', 'all']),
              help='Street network type (default: all)')
@click.option('--output', '-o', default='output',
              help='Output directory (default: output)')
@click.option('--cache/--no-cache', default=True,
              help='Use cache for downloaded data (default: True)')
@click.option('--verbose', '-v', is_flag=True,
              help='Show detailed progress information')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress progress bars')
def analyze(
    latitude: float,
    longitude: float,
    radius: int,
    network_type: str,
    output: str,
    cache: bool,
    verbose: bool,
    quiet: bool
):
    """Analyze urban features for a location.
    
    Example:
        geofeaturekit analyze 40.7580 -73.9855 --radius 500
        geofeaturekit analyze 40.7580 -73.9855 --radius 500 --verbose
        geofeaturekit analyze 40.7580 -73.9855 --radius 500 --quiet
    """
    # Create output directory
    os.makedirs(output, exist_ok=True)
    
    # Determine progress settings
    show_progress = not quiet
    progress_detail = 'verbose' if verbose else 'normal'
    
    # Initialize extractor with progress settings
    extractor = UrbanFeatureExtractor(
        radius_meters=radius,
        use_cache=cache,
        show_progress=show_progress,
        progress_detail=progress_detail
    )
    
    # Extract features
    if not quiet:
        click.echo(f"Analyzing location ({latitude}, {longitude}) with {radius}m radius...")
    
    try:
        results = extractor.features_from_location(
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius
        )
        
        # Save results
        output_file = os.path.join(
            output,
            f"analysis_{latitude:.4f}_{longitude:.4f}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not quiet:
            click.echo(f"Results saved to {output_file}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

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
@click.option('--verbose', '-v', is_flag=True,
              help='Show detailed progress information')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress progress bars')
def batch_analyze(
    input_file: str,
    radius: int,
    network_type: str,
    output: str,
    cache: bool,
    verbose: bool,
    quiet: bool
):
    """Analyze multiple locations from a JSON file.
    
    The input file should contain a list of objects with 'latitude'
    and 'longitude' fields.
    
    Example:
        geofeaturekit batch-analyze locations.json --radius 500
        geofeaturekit batch-analyze locations.json --radius 500 --verbose
        geofeaturekit batch-analyze locations.json --radius 500 --quiet
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
            if not quiet:
                click.echo(f"Warning: Skipping invalid location {loc}")
    
    if not location_tuples:
        raise click.BadParameter("No valid locations found in input file")
    
    # Determine progress settings
    show_progress = not quiet
    progress_detail = 'verbose' if verbose else 'normal'
    
    # Initialize extractor with progress settings
    extractor = UrbanFeatureExtractor(
        radius_meters=radius,
        use_cache=cache,
        show_progress=show_progress,
        progress_detail=progress_detail
    )
    
    # Extract features
    if not quiet:
        click.echo(f"Analyzing {len(location_tuples)} locations with {radius}m radius...")
    
    try:
        results = extractor.features_from_location_batch(
            locations=location_tuples,
            radius_meters=radius
        )
        
        # Save results
        output_file = os.path.join(output, "batch_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not quiet:
            successful_count = len([r for r in results if r is not None])
            click.echo(f"Batch analysis complete: {successful_count}/{len(location_tuples)} locations processed successfully")
            click.echo(f"Results saved to {output_file}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli() 