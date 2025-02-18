"""
PyGreedy Utils Module
===================

This module provides utility functions and helpers for the PyGreedy package,
including image I/O, transformation utilities, visualization tools, and logging.

Created by: devhliu
Created at: 2025-02-18 04:34:13 UTC
"""

from .image_io import (
    load_image,
    save_image,
    load_nifti,
    save_nifti,
    load_dicom_series,
    save_dicom_series
)

from .transform import (
    compute_grid,
    compose_transforms,
    invert_transform,
    interpolate_deformation,
    compute_jacobian
)

from .visualization import (
    plot_registration_result,
    plot_deformation_field,
    plot_metric_history,
    create_checkerboard,
    overlay_images
)

from .logger import (
    setup_logger,
    get_logger,
    log_registration_params,
    log_registration_result
)

# Version of the utils module
__version__ = "0.1.0"

# Module exports
__all__ = [
    # Image I/O
    'load_image',
    'save_image',
    'load_nifti',
    'save_nifti',
    'load_dicom_series',
    'save_dicom_series',
    
    # Transform utilities
    'compute_grid',
    'compose_transforms',
    'invert_transform',
    'interpolate_deformation',
    'compute_jacobian',
    
    # Visualization
    'plot_registration_result',
    'plot_deformation_field',
    'plot_metric_history',
    'create_checkerboard',
    'overlay_images',
    
    # Logging
    'setup_logger',
    'get_logger',
    'log_registration_params',
    'log_registration_result'
]

# Module metadata
__author__ = "devhliu"
__created_at__ = "2025-02-18 04:34:13"
__module_name__ = "pygreedy.core.utils"

def get_module_info():
    """
    Get information about the utils module.

    Returns
    -------
    dict
        Dictionary containing module metadata
    """
    return {
        'name': __module_name__,
        'version': __version__,
        'author': __author__,
        'created_at': __created_at__,
        'description': 'Utility functions for PyGreedy registration',
    }

# Initialize logger
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def test_dependencies():
    """
    Test if all required dependencies are available.
    
    Returns
    -------
    bool
        True if all dependencies are available
    
    Raises
    ------
    ImportError
        If any required dependency is missing
    """
    required_packages = [
        'numpy',
        'nibabel',
        'pydicom',
        'matplotlib',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}"
        )
    
    return True

# Test dependencies on import
try:
    test_dependencies()
except ImportError as e:
    logger.warning(f"Dependency check failed: {str(e)}")

# Optional GPU check
import torch
if torch.cuda.is_available():
    logger.info("GPU support detected and enabled")
else:
    logger.info("No GPU support detected, using CPU only")