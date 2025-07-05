"""Progress tracking utilities."""

import logging
from typing import Optional, Iterator, TypeVar, Sequence, Iterable, Any
from tqdm import tqdm

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