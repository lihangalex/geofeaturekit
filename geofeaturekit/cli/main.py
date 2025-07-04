"""Command line interface for GeoFeatureKit."""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd

from ..core.config import Config, AnalysisConfig, DEFAULT_OUTPUT_DIR
from ..core.models import Location
from ..core.extractor import GeospatialFeatureExtractor
from ..exceptions.errors import GeoFeatureKitError

def setup_logging(verbose: bool):
    """Configure logging based on verbosity.
    
    Args:
        verbose: Whether to enable debug logging
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract geospatial features from OpenStreetMap data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single location
    geofeaturekit --lat 43.6532 --lon -79.3832 --name "CN Tower"
    
    # Analyze multiple locations from CSV
    geofeaturekit --input locations.csv
    
    # Customize analysis area and save to disk
    geofeaturekit --lat 43.6532 --lon -79.3832 --radius 1000 --save-dir ./results
    """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=str,
        help='CSV file containing locations (must have latitude,longitude columns)'
    )
    input_group.add_argument(
        '--lat',
        type=float,
        help='Latitude of location'
    )
    
    # Only require longitude if latitude is provided
    parser.add_argument(
        '--lon',
        type=float,
        help='Longitude of location'
    )
    
    # Optional arguments
    parser.add_argument(
        '--name',
        type=str,
        help='Name of location (optional)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=300,  # Updated to match DEFAULT_RADIUS_METERS
        help='Analysis radius in meters (default: 300)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save results (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--in-memory',
        action='store_true',
        help='Keep results in memory instead of saving to disk'
    )
    parser.add_argument(
        '--user-agent',
        type=str,
        help='User agent for API requests (default: geofeaturekit)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable OSM data caching'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.lat is not None and args.lon is None:
        parser.error("--lon is required when --lat is provided")
    
    return args

def main():
    """Main CLI entry point."""
    try:
        args = parse_args()
        setup_logging(args.verbose)
        
        # Create configurations
        config = Config(
            user_agent=args.user_agent or "geofeaturekit",
            cache_enabled=not args.no_cache,
            log_to_console=not args.quiet,
            save_dir=None if args.in_memory else args.save_dir
        )
        
        analysis_config = AnalysisConfig(
            radius_meters=args.radius,
            output_dir=None if args.in_memory else args.save_dir
        )
        
        # Initialize extractor
        extractor = GeospatialFeatureExtractor(config=analysis_config)
        
        # Process input
        if args.input:
            # Read locations from CSV
            try:
                df = pd.read_csv(args.input)
                required_cols = ['latitude', 'longitude']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(
                        f"Input CSV must contain columns: {', '.join(required_cols)}"
                    )
                locations = df[required_cols].to_dict('records')
            except Exception as e:
                raise GeoFeatureKitError(f"Error reading input file: {str(e)}")
        else:
            # Single location
            locations = {
                "latitude": args.lat,
                "longitude": args.lon
            }
        
        # Extract features
        results = extractor.extract_features(locations)
        
        # Output results
        if results is not None:
            if isinstance(results, list):
                # Convert list of results to DataFrame for display
                df = pd.DataFrame([r.to_dict() for r in results])
                print(df.to_string())
            else:
                # Single result
                print(results.to_dict())
        else:
            print(f"\nResults saved to: {args.save_dir}")
        
        return 0
        
    except GeoFeatureKitError as e:
        logging.error(str(e))
        return 1
    except Exception as e:
        logging.exception("Unexpected error")
        return 2

if __name__ == '__main__':
    sys.exit(main()) 