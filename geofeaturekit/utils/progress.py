"""Progress tracking utilities."""

import logging
import time
from typing import Optional, Iterator, TypeVar, Sequence, Iterable, Any, Dict
from tqdm import tqdm
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

def create_progress_bar(
    iterable: Sequence[T],
    desc: str = "Processing",
    total: Optional[int] = None,
    disable: bool = False
) -> Iterator[T]:
    """Create a progress bar for iteration.
    
    Args:
        iterable: Sequence to iterate over
        desc: Description for the progress bar
        total: Total number of items (defaults to len(iterable))
        disable: Whether to disable the progress bar
        
    Returns:
        Iterator with progress bar
    """
    if total is None:
        total = len(iterable)
    
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=disable,
        unit="loc",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

def log_analysis_start(location_count: int):
    """Log the start of analysis.
    
    Args:
        location_count: Number of locations to analyze
    """
    logger.info(f"Starting analysis of {location_count} location(s)")

def log_analysis_complete(location_count: int, success_count: int):
    """Log analysis completion.
    
    Args:
        location_count: Total number of locations
        success_count: Number of successfully analyzed locations
    """
    logger.info(
        f"Analysis complete. Successfully processed {success_count}/{location_count} "
        f"location(s) ({(success_count/location_count)*100:.1f}%)"
    )

def log_error(location: str, error: Exception):
    """Log an error during analysis.
    
    Args:
        location: Location identifier (name or coordinates)
        error: Exception that occurred
    """
    logger.error(f"Error processing location {location}: {str(error)}")

def log_rate_limit():
    """Log rate limit sleep."""
    logger.debug("Sleeping to respect rate limits...")

class ProgressTracker:
    """Track progress of feature extraction."""
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.current_step = None
        self.total_steps = 0
        self.completed_steps = 0
    
    def start(self, message: str):
        """Start tracking a new step.
        
        Args:
            message: Description of the step
        """
        self.current_step = message
        logger.info(f"Starting: {message}")
    
    def update(self, message: str):
        """Update the current step.
        
        Args:
            message: New status message
        """
        self.current_step = message
        logger.info(message)
    
    def complete(self):
        """Mark the current step as complete."""
        if self.current_step:
            logger.info(f"Completed: {self.current_step}")
            self.current_step = None
            self.completed_steps += 1

class EnhancedProgressTracker:
    """Enhanced progress tracker with phase-based progress and time estimation."""
    
    def __init__(self, show_progress: bool = True, detail_level: str = 'normal'):
        """Initialize the enhanced progress tracker.
        
        Args:
            show_progress: Whether to show progress bars
            detail_level: Level of detail ('minimal', 'normal', 'verbose')
        """
        self.show_progress = show_progress
        self.detail_level = detail_level
        self.start_time = None
        self.current_phase = None
        self.phase_weights = {
            'download': 0.7,
            'calculate': 0.25,
            'format': 0.05
        }
        self.main_pbar = None
        self.sub_pbar = None
        self.completed_weight = 0.0
        
    def start_extraction(self, location_desc: str = ""):
        """Start the feature extraction process.
        
        Args:
            location_desc: Description of the location being processed
        """
        self.start_time = time.time()
        if self.show_progress:
            desc = f"Extracting Features"
            if location_desc:
                desc += f" - {location_desc}"
            self.main_pbar = tqdm(
                total=100,
                desc=desc,
                unit="%",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]"
            )
    
    @contextmanager
    def phase(self, phase_name: str, description: str = ""):
        """Context manager for tracking a phase of extraction.
        
        Args:
            phase_name: Name of the phase ('download', 'calculate', 'format')
            description: Optional description for the phase
        """
        self.current_phase = phase_name
        phase_weight = self.phase_weights.get(phase_name, 0.1)
        
        # Update main progress bar description
        if self.main_pbar:
            phase_desc = description or phase_name.title()
            self.main_pbar.set_description(f"Extracting Features - {phase_desc}")
        
        # Create sub-progress bar if detailed progress is enabled
        if self.show_progress and self.detail_level in ['normal', 'verbose']:
            self.sub_pbar = tqdm(
                total=100,
                desc=f"  {description or phase_name.title()}",
                unit="%",
                ncols=80,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}]"
            )
        
        try:
            yield self
        finally:
            # Close sub-progress bar
            if self.sub_pbar:
                self.sub_pbar.close()
                self.sub_pbar = None
            
            # Update main progress bar
            if self.main_pbar:
                self.completed_weight += phase_weight
                self.main_pbar.update(int(phase_weight * 100))
    
    def update_phase_progress(self, percent: float, description: str = ""):
        """Update progress within the current phase.
        
        Args:
            percent: Percentage complete within the phase (0-100)
            description: Optional description of current step
        """
        if self.sub_pbar:
            if description:
                self.sub_pbar.set_description(f"  {description}")
            # Update to absolute percentage
            current_progress = self.sub_pbar.n
            self.sub_pbar.update(percent - current_progress)
    
    def complete_extraction(self):
        """Complete the feature extraction process."""
        if self.main_pbar:
            # Ensure we're at 100%
            remaining = 100 - self.main_pbar.n
            if remaining > 0:
                self.main_pbar.update(remaining)
            
            # Show completion message
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            self.main_pbar.set_description(f"Extraction Complete ({elapsed_time:.1f}s)")
            self.main_pbar.close()
            self.main_pbar = None
    
    def error(self, message: str):
        """Handle an error during extraction.
        
        Args:
            message: Error message
        """
        if self.main_pbar:
            self.main_pbar.set_description(f"Error: {message}")
            self.main_pbar.close()
            self.main_pbar = None
        if self.sub_pbar:
            self.sub_pbar.close()
            self.sub_pbar = None

class BatchProgressTracker:
    """Progress tracker for batch operations."""
    
    def __init__(self, total_locations: int, show_progress: bool = True):
        """Initialize batch progress tracker.
        
        Args:
            total_locations: Total number of locations to process
            show_progress: Whether to show progress bars
        """
        self.total_locations = total_locations
        self.show_progress = show_progress
        self.current_location = 0
        self.successful_locations = 0
        self.failed_locations = 0
        self.start_time = None
        self.main_pbar = None
        
    def start_batch(self):
        """Start batch processing."""
        self.start_time = time.time()
        if self.show_progress:
            self.main_pbar = tqdm(
                total=self.total_locations,
                desc="Processing Locations",
                unit="loc",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
    
    def start_location(self, location_desc: str = ""):
        """Start processing a location.
        
        Args:
            location_desc: Description of the current location
        """
        self.current_location += 1
        if self.main_pbar:
            desc = f"Processing location {self.current_location}/{self.total_locations}"
            if location_desc:
                desc += f" - {location_desc}"
            self.main_pbar.set_description(desc)
    
    def complete_location(self, success: bool = True):
        """Complete processing a location.
        
        Args:
            success: Whether the location was processed successfully
        """
        if success:
            self.successful_locations += 1
        else:
            self.failed_locations += 1
        
        if self.main_pbar:
            self.main_pbar.update(1)
            
            # Update postfix with success/failure counts
            postfix = {
                'Success': self.successful_locations,
                'Failed': self.failed_locations
            }
            self.main_pbar.set_postfix(postfix)
    
    def complete_batch(self):
        """Complete batch processing."""
        if self.main_pbar:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            success_rate = (self.successful_locations / self.total_locations) * 100
            self.main_pbar.set_description(
                f"Batch Complete - {self.successful_locations}/{self.total_locations} "
                f"({success_rate:.1f}%) in {elapsed_time:.1f}s"
            )
            self.main_pbar.close()
            self.main_pbar = None 