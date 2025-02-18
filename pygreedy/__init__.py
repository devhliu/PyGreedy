"""
PyGreedy: Python Implementation of Greedy Diffeomorphic Registration
=================================================================

PyGreedy is a Python package for fast diffeomorphic image registration,
based on the original C++ implementation by Paul Yushkevich.

Main Features
------------
* Fast diffeomorphic image registration
* Both affine and deformable registration support
* Multiple similarity metrics
* Multi-resolution optimization
* GPU acceleration through PyTorch
* User-friendly Python interface

Example
-------
>>> from pygreedy import GreedyRegistration
>>> from pygreedy.utils import load_image, save_image
>>> 
>>> # Load images
>>> fixed_data, fixed_affine = load_image("fixed.nii.gz")
>>> moving_data, moving_affine = load_image("moving.nii.gz")
>>> 
>>> # Create registration object
>>> reg = GreedyRegistration(iterations=200, sigma=2.0)
>>> 
>>> # Perform registration
>>> result = reg.register(fixed_data, moving_data)
"""

__version__ = "0.1.0"
__author__ = "devhliu"
__email__ = "huiliu.liu@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 devhliu"
__created_at__ = "2025-02-18"

import os
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
from .core.registration import GreedyRegistration
from .core.affine import AffineRegistration
from .core.diffeomorphic import DiffeomorphicRegistration
from .core.optimizer import RegistrationOptimizer
from .core.metrics import (
    NormalizedCrossCorrelation,
    MutualInformation,
    MeanSquaredError
)

# Import utilities
from .utils.image_io import load_image, save_image
from .utils.transform import compose_transforms, compute_grid
from .utils.visualization import plot_registration_result

# Package configuration
_config: Dict[str, Any] = {
    'use_gpu': True,
    'default_metric': 'ncc',
    'verbose': False,
    'data_dir': os.path.join(os.path.dirname(__file__), '..', 'data'),
}

def configure(
    use_gpu: Optional[bool] = None,
    default_metric: Optional[str] = None,
    verbose: Optional[bool] = None,
    data_dir: Optional[str] = None
) -> None:
    """
    Configure global PyGreedy settings.

    Parameters
    ----------
    use_gpu : bool, optional
        Whether to use GPU acceleration when available
    default_metric : str, optional
        Default similarity metric ('ncc', 'mse', or 'mi')
    verbose : bool, optional
        Whether to print detailed progress information
    data_dir : str, optional
        Directory for storing temporary and cached data
    """
    if use_gpu is not None:
        _config['use_gpu'] = use_gpu
    if default_metric is not None:
        _config['default_metric'] = default_metric
    if verbose is not None:
        _config['verbose'] = verbose
    if data_dir is not None:
        _config['data_dir'] = data_dir

def get_config() -> Dict[str, Any]:
    """
    Get current PyGreedy configuration.

    Returns
    -------
    dict
        Current configuration settings
    """
    return _config.copy()

# Define package exports
__all__ = [
    # Core classes
    'GreedyRegistration',
    'AffineRegistration',
    'DiffeomorphicRegistration',
    'RegistrationOptimizer',
    
    # Metrics
    'NormalizedCrossCorrelation',
    'MutualInformation',
    'MeanSquaredError',
    
    # Utilities
    'load_image',
    'save_image',
    'compose_transforms',
    'compute_grid',
    'plot_registration_result',
    
    # Configuration
    'configure',
    'get_config',
]

# Check for GPU availability
import torch
if _config['use_gpu'] and not torch.cuda.is_available():
    logger.warning(
        "GPU acceleration requested but no CUDA device available. "
        "Falling back to CPU computation."
    )
    _config['use_gpu'] = False

# Version information
__version_info__ = tuple(map(int, __version__.split('.')))

def show_versions() -> None:
    """Print version information for PyGreedy and its dependencies."""
    import numpy
    import scipy
    import torch
    import nibabel
    
    print(f"PyGreedy: {__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Nibabel: {nibabel.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")