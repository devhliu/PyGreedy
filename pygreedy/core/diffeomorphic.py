"""
PyGreedy Diffeomorphic Registration Module
========================================

This module implements diffeomorphic image registration using a velocity field
parameterization and scaling-and-squaring for deformation field computation.

Created by: devhliu
Created at: 2025-02-18 04:21:43 UTC
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path

from .metrics import SimilarityMetric, NormalizedCrossCorrelation
from .optimizer import RegistrationOptimizer
from .parameters import RegistrationParameters
from ..utils.transform import compute_grid
from ..utils.image_io import load_image, save_image

class VelocityField:
    """
    Velocity field parameterization for diffeomorphic transformations.
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        scaling_steps: int = 6,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize velocity field parameterization.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the image domain
        scaling_steps : int, optional
            Number of scaling and squaring steps, by default 6
        device : torch.device, optional
            Device for computations, by default CPU
        """
        self.shape = shape
        self.scaling_steps = scaling_steps
        self.device = device
        
    def compute_deformation(
        self,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute deformation field from velocity field using scaling and squaring.

        Parameters
        ----------
        velocity : torch.Tensor
            Velocity field tensor

        Returns
        -------
        torch.Tensor
            Deformation field
        """
        # Scale velocity field
        scaled_velocity = velocity / (2 ** self.scaling_steps)
        
        # Initialize deformation field
        phi = scaled_velocity
        
        # Scaling and squaring steps
        for _ in range(self.scaling_steps):
            phi = self._compose_deformations(phi, phi)
            
        return phi
    
    def _compose_deformations(
        self,
        phi1: torch.Tensor,
        phi2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compose two deformation fields.

        Parameters
        ----------
        phi1 : torch.Tensor
            First deformation field
        phi2 : torch.Tensor
            Second deformation field

        Returns
        -------
        torch.Tensor
            Composed deformation field
        """
        # Create sampling grid
        grid = compute_grid(phi2)
        
        # Warp first deformation with second
        warped = torch.nn.functional.grid_sample(
            phi1.permute(0, 2, 3, 1).unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(0).permute(2, 0, 1)
        
        return warped + phi2

class DiffeomorphicRegistration:
    """
    Diffeomorphic registration using velocity field parameterization.
    """
    
    def __init__(
        self,
        iterations: int = 200,
        learning_rate: float = 0.1,
        scaling_steps: int = 6,
        regularization_weight: float = 0.1,
        metric: str = "ncc",
        optimizer_type: str = "adam",
        use_gpu: bool = True,
        verbose: bool = False,
        parameters: Optional[RegistrationParameters] = None
    ):
        """
        Initialize diffeomorphic registration.

        Parameters
        ----------
        iterations : int, optional
            Number of optimization iterations, by default 200
        learning_rate : float, optional
            Learning rate for optimization, by default 0.1
        scaling_steps : int, optional
            Number of scaling and squaring steps, by default 6
        regularization_weight : float, optional
            Weight of regularization term, by default 0.1
        metric : str, optional
            Similarity metric ('ncc', 'mse', or 'mi'), by default "ncc"
        optimizer_type : str, optional
            Type of optimizer ('adam', 'sgd', or 'lbfgs'), by default "adam"
        use_gpu : bool, optional
            Whether to use GPU acceleration, by default True
        verbose : bool, optional
            Whether to print progress, by default False
        parameters : RegistrationParameters, optional
            Additional registration parameters
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.parameters = parameters or RegistrationParameters()
        self.scaling_steps = scaling_steps
        self.regularization_weight = regularization_weight
        
        # Initialize optimizer
        self.optimizer = RegistrationOptimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            num_iterations=[iterations],
            use_gpu=use_gpu,
            verbose=verbose
        )
        
        # Initialize metric
        self.metric = NormalizedCrossCorrelation()
        
    def register(
        self,
        fixed: Union[np.ndarray, str, Path],
        moving: Union[np.ndarray, str, Path],
        initial_velocity: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform diffeomorphic registration.

        Parameters
        ----------
        fixed : Union[np.ndarray, str, Path]
            Fixed image or path to fixed image
        moving : Union[np.ndarray, str, Path]
            Moving image or path to moving image
        initial_velocity : Optional[np.ndarray], optional
            Initial velocity field, by default None
        mask : Optional[np.ndarray], optional
            Mask for the fixed image, by default None

        Returns
        -------
        Dict[str, Any]
            Registration results including deformation field and warped image
        """
        # Load images if necessary
        fixed_data, fixed_affine = self._load_image_data(fixed)
        moving_data, moving_affine = self._load_image_data(moving)
        
        # Convert to torch tensors
        fixed_tensor = torch.from_numpy(fixed_data).float().to(self.device)
        moving_tensor = torch.from_numpy(moving_data).float().to(self.device)
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().to(self.device)
        else:
            mask_tensor = None
            
        # Initialize velocity field
        velocity_field = VelocityField(
            fixed_data.shape,
            scaling_steps=self.scaling_steps,
            device=self.device
        )
        
        if initial_velocity is None:
            velocity = torch.zeros(
                (fixed_data.ndim, *fixed_data.shape),
                device=self.device,
                requires_grad=True
            )
        else:
            velocity = torch.from_numpy(initial_velocity).to(self.device)
            velocity.requires_grad_(True)
            
        # Optimize
        result = self.optimizer.optimize(
            fixed_tensor,
            moving_tensor,
            lambda x, v: self._transform_image(x, v, velocity_field),
            lambda x, y: self._compute_loss(x, y, mask_tensor, velocity),
            velocity
        )
        
        # Compute final deformation
        final_deformation = velocity_field.compute_deformation(
            result['final_parameters']
        )
        
        return {
            'velocity_field': result['final_parameters'].detach().cpu().numpy(),
            'deformation_field': final_deformation.detach().cpu().numpy(),
            'warped_image': self._transform_image(
                moving_tensor,
                result['final_parameters'],
                velocity_field
            ).detach().cpu().numpy(),
            'convergence': result['convergence_achieved'],
            'fixed_affine': fixed_affine,
            'moving_affine': moving_affine
        }
    
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
        velocity: torch.Tensor,
        velocity_field: VelocityField
    ) -> torch.Tensor:
        """Apply transformation to image using velocity field."""
        deformation = velocity_field.compute_deformation(velocity)
        grid = compute_grid(deformation)
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
        mask: Optional[torch.Tensor],
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute registration loss with regularization.

        Parameters
        ----------
        fixed : torch.Tensor
            Fixed image
        moving : torch.Tensor
            Moving image
        mask : Optional[torch.Tensor]
            Mask for the fixed image
        velocity : torch.Tensor
            Current velocity field

        Returns
        -------
        torch.Tensor
            Total loss value
        """
        similarity = self.metric(fixed, moving)
        if mask is not None:
            similarity = similarity * mask
            
        regularization = self.regularization_weight * self._compute_regularization(velocity)
        
        return similarity + regularization
    
    def _compute_regularization(
        self,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regularization term (smoothness of velocity field).

        Parameters
        ----------
        velocity : torch.Tensor
            Velocity field

        Returns
        -------
        torch.Tensor
            Regularization term value
        """
        # Compute gradients
        gradients = torch.gradient(velocity)
        
        # Compute L2 norm of gradients
        reg = sum(torch.sum(g**2) for g in gradients)
        
        return reg