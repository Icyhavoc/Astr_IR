"""Source-masked two-dimensional background subtraction."""

from .processor import BackgroundConfig, run_batch, subtract_background

__all__ = ["BackgroundConfig", "run_batch", "subtract_background"]
