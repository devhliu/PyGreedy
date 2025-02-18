"""
PyGreedy Core Module
===================

This module provides the core functionality for the PyGreedy package,
including registration algorithms, optimizers, and metrics.

Components
---------
- GreedyRegistration: Main registration class
- AffineRegistration: Affine transformation registration
- DiffeomorphicRegistration: Diffeomorphic registration
- RegistrationOptimizer: Optimization strategies
- Various similarity metrics

Created by: devhliu
Created at: 2025-02-18 04:13:14 UTC
"""

from typing import Dict, Any, Type

from .registration import GreedyRegistration
from .affine import AffineRegistration
from .diffeomorphic import DiffeomorphicRegistration
from .optimizer import RegistrationOptimizer
from .metrics import (
    SimilarityMetric,
    NormalizedCrossCorrelation,
    MutualInformation,
    MeanSquaredError
)
from .parameters import RegistrationParameters

# Version of the core module
__version__ = "0.1.0"

# Mapping of metric names to their implementations
AVAILABLE_METRICS: Dict[str, Type[SimilarityMetric]] = {
    'ncc': NormalizedCrossCorrelation,
    'mi': MutualInformation,
    'mse': MeanSquaredError,
}

# Mapping of registration types to their implementations
REGISTRATION_TYPES: Dict[str, Any] = {
    'affine': AffineRegistration,
    'diffeomorphic': DiffeomorphicRegistration,
}

def get_metric(metric_name: str) -> Type[SimilarityMetric]:
    """
    Get similarity metric class by name.

    Parameters
    ----------
    metric_name : str
        Name of the metric ('ncc', 'mi', or 'mse')

    Returns
    -------
    Type[SimilarityMetric]
        Metric class implementation

    Raises
    ------
    ValueError
        If metric_name is not recognized
    """
    metric_name = metric_name.lower()
    if metric_name not in AVAILABLE_METRICS:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Available metrics: {list(AVAILABLE_METRICS.keys())}"
        )
    return AVAILABLE_METRICS[metric_name]

def get_registration_type(reg_type: str) -> Any:
    """
    Get registration class by type.

    Parameters
    ----------
    reg_type : str
        Type of registration ('affine' or 'diffeomorphic')

    Returns
    -------
    Type
        Registration class implementation

    Raises
    ------
    ValueError
        If reg_type is not recognized
    """
    reg_type = reg_type.lower()
    if reg_type not in REGISTRATION_TYPES:
        raise ValueError(
            f"Unknown registration type: {reg_type}. "
            f"Available types: {list(REGISTRATION_TYPES.keys())}"
        )
    return REGISTRATION_TYPES[reg_type]

# Module exports
__all__ = [
    # Main classes
    'GreedyRegistration',
    'AffineRegistration',
    'DiffeomorphicRegistration',
    'RegistrationOptimizer',
    'RegistrationParameters',
    
    # Metrics
    'SimilarityMetric',
    'NormalizedCrossCorrelation',
    'MutualInformation',
    'MeanSquaredError',
    
    # Helper functions
    'get_metric',
    'get_registration_type',
    
    # Constants
    'AVAILABLE_METRICS',
    'REGISTRATION_TYPES',
]

# Module metadata
__author__ = "devhliu"
__created_at__ = "2025-02-18 04:13:14"
__module_name__ = "pygreedy.core"

def get_module_info() -> Dict[str, str]:
    """
    Get information about the core module.

    Returns
    -------
    Dict[str, str]
        Dictionary containing module metadata
    """
    return {
        'name': __module_name__,
        'version': __version__,
        'author': __author__,
        'created_at': __created_at__,
        'description': 'Core functionality for PyGreedy registration',
    }

# Additional initialization if needed
import logging

logger = logging.getLogger(__name__)
logger.info(f"Initializing PyGreedy core module v{__version__}")