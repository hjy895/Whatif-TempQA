"""
Logging configuration and progress tracking utilities.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure root logger for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress verbose third-party library output
    for lib in ('transformers', 'torch', 'urllib3'):
        logging.getLogger(lib).setLevel(logging.WARNING)


class ProgressLogger:
    """Tracks and logs pipeline progress at regular intervals."""

    def __init__(self, name: str, total: int):
        self.logger = logging.getLogger(name)
        self.total = total
        self.current = 0
        self.start_time = datetime.now()

    def update(self, increment: int = 1):
        self.current += increment
        if self.current % 1000 == 0:
            elapsed = datetime.now() - self.start_time
            rate = self.current / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            self.logger.info(
                f"Progress: {self.current}/{self.total} "
                f"({100 * self.current / self.total:.1f}%) "
                f"| Rate: {rate:.0f}/s "
                f"| ETA: {remaining:.0f}s"
            )
