"""
Logging Utilities
"""

import logging
import sys
from typing import Optional

def setup_logger(name: str = "BRD_Generator", level: int = logging.INFO) -> logging.Logger:
    """Setup and configure logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance"""
    if name is None:
        name = "BRD_Generator"
    return logging.getLogger(name)
