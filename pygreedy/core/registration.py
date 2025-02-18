"""
PyGreedy Core Registration Module
===============================

This module implements the main registration functionality for PyGreedy.

Created by: devhliu
Created at: 2025-02-18 04:15:54 UTC
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path

from .optimizer import RegistrationOptimizer
from .metrics import NormalizedCrossCorrelation, MutualInformation, MeanSquaredError
from .parameters import RegistrationParameters
from ..utils.transform import compute_grid, compose_transforms
from ..utils.image_io import load_image, save_image

class GreedyRegistration:
    """
    Main registration class implementing greedy diffeomorphic registration.
    
    This class provides a unified interface for both affine and deformable
    registration with support for various similarity metrics and optimization
    strategies.
    """
    
    def __init__(
        self,
        iterations: int = 100,
        sigma: float = 3.0,
        learning_rate: float = 0.1,
        metric: str = "ncc",
        use_gpu: bool = True,
        verbose: bool = False,
        parameters: Optional[RegistrationParameters] = None
    ):
        """
        Initialize registration parameters.
        
        Parameters
        ----------
        iterations : int, optional
            Number of optimization iterations, by default 100
        sigma : float, optional
            Gaussian smoothing sigma, by default 3.0
        learning_rate : float, optional
            Gradient descent learning rate, by default 0.1
        metric : str, optional
            Similarity metric ('ncc', 'mse', or 'mi'), by default "ncc"
        use_gpu : bool, optional
            Whether to use GPU acceleration, by default True
        verbose : bool, optional
            Whether to print progress, by default False
        parameters : RegistrationParameters, optional
            Additional registration parameters
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.parameters = parameters or RegistrationParameters()
        self.parameters.update({
            'iterations': iterations,
            'sigma': sigma,
            'learning_rate': learning_rate,
            'metric': metric,
            'verbose': verbose
        })
        
        # Initialize metric
        self.metric = self._initialize_metric(metric)
        
        # Initialize optimizer
        self.optimizer = RegistrationOptimizer(
            learning_rate=learning_rate,
            num_levels=self.parameters.num_levels,
            scale_factor=self.parameters.scale_factor,
            num_iterations=[iterations] * self.parameters.num_levels,
            smoothing_sigmas=[sigma] * self.parameters.num_levels,
            use_gpu=use_gpu,
            verbose=verbose
        )
        
    def register(
        self,
        fixed: Union[np.ndarray, str, Path],
        moving: Union[np.ndarray, str, Path],
        initial_transform: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Register moving image to fixed image.
        
        Parameters
        ----------
        fixed : Union[np.ndarray, str, Path]
            Fixed (target) image or path to image file
        moving : Union[np.ndarray, str, Path]
            Moving (source) image or path to image file
        initial_transform : Optional[np.ndarray], optional
            Initial transformation matrix, by default None
        mask : Optional[np.ndarray], optional
            Mask for the fixed image, by default None
            
        Returns
        -------
        Dict[str, Any]
            Registration results including:
            - transform: Final transformation
            - warped_image: Transformed moving image
            - displacement_field: Dense displacement field
            - metrics: Registration quality metrics
        """
        # Load images if paths provided
        fixed_data, fixed_affine = self._load_image_data(fixed)
        moving_data, moving_affine = self._load_image_data(moving)
        
        # Convert to torch tensors
        fixed_tensor = torch.from_numpy(fixed_data).float().to(self.device)
        moving_tensor = torch.from_numpy(moving_data).float().to(self.device)
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().to(self.device)
        else:
            mask_tensor = None
            
        # Initialize transformation
        if initial_transform is None:
            phi = torch.zeros(
                (1, fixed_data.ndim, *fixed_data.shape),
                device=self.device,
                requires_grad=True
            )
        else:
            phi = torch.from_numpy(initial_transform).to(self.device)
            phi.requires_grad = True
            
        # Perform optimization
        result = self.optimizer.optimize(
            fixed_tensor,
            moving_tensor,
            lambda x, p: self._transform_image(x, p),
            lambda x, y: self._compute_loss(x, y, mask_tensor),
            phi
        )
        
        # Compute final metrics
        final_metrics = self._compute_registration_metrics(
            fixed_tensor,
            moving_tensor,
            result['final_parameters']
        )
        
        # Prepare results
        return {
            'transform': result['final_parameters'].detach().cpu().numpy(),
            'warped_image': self._transform_image(
                moving_tensor,
                result['final_parameters']
            ).detach().cpu().numpy(),
            'displacement_field': compute_grid(
                result['final_parameters']
            ).detach().cpu().numpy(),
            'metrics': final_metrics,
            'convergence': result['convergence_achieved'],
            'iterations_per_level': result.get('iterations_per_level', []),
            'fixed_affine': fixed_affine,
            'moving_affine': moving_affine
        }
    
    def _initialize_metric(self, metric_name: str) -> Any:
        """Initialize similarity metric."""
        metric_map = {
            'ncc': NormalizedCrossCorrelation(),
            'mse': MeanSquaredError(),
            'mi': MutualInformation()
        }
        if metric_name.lower() not in metric_map:
            raise ValueError(f"Unknown metric: {metric_name}")
        return metric_map[metric_name.lower()]
    
    def _load_image_data(
        self,
        image: Union[np.ndarray, str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load image data from array or file."""
        if isinstance(image, (str, Path)):
            return load_image(str(image))
        elif isinstance(image, np.ndarray):
            return image, np.eye(4)
        else:
            raise TypeError("Image must be numpy array or path to image file")
    
    def _transform_image(
        self,
        image: torch.Tensor,
        transform: torch.Tensor
    ) -> torch.Tensor:
        """Apply transformation to image."""
        grid = compute_grid(transform)
        return torch.nn.functional.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze()
    
    def _compute_loss(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute registration loss."""
        similarity = self.metric(fixed, moving)
        if mask is not None:
            similarity = similarity * mask
        regularization = self.parameters.regularization_weight * self._compute_regularization()
        return similarity + regularization
    
    def _compute_regularization(self) -> torch.Tensor:
        """Compute regularization term."""
        # Implement regularization computation
        return torch.tensor(0.0, device=self.device)
    
    def _compute_registration_metrics(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        transform: torch.Tensor
    ) -> Dict[str, float]:
        """Compute final registration quality metrics."""
        warped = self._transform_image(moving, transform)
        return {
            'final_loss': self._compute_loss(fixed, warped).item(),
            'ncc': NormalizedCrossCorrelation()(fixed, warped).item(),
            'mse': MeanSquaredError()(fixed, warped).item(),
            'mi': MutualInformation()(fixed, warped).item()
        }
    
    def save_transform(
        self,
        filename: Union[str, Path],
        transform: np.ndarray
    ) -> None:
        """Save transformation to file."""
        np.save(str(filename), transform)
    
    def load_transform(
        self,
        filename: Union[str, Path]
    ) -> np.ndarray:
        """Load transformation from file."""
        return np.load(str(filename))