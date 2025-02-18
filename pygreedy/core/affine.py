"""
PyGreedy Affine Registration Module
=================================

This module implements affine registration functionality, supporting various
transformation models including rigid, similarity, and full affine transformations.

Created by: devhliu
Created at: 2025-02-18 04:18:49 UTC
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

class AffineTransform:
    """
    Affine transformation parameterization and operations.
    """
    
    def __init__(
        self,
        dim: int = 3,
        transform_type: str = "affine"
    ):
        """
        Initialize affine transformation.

        Parameters
        ----------
        dim : int, optional
            Dimensionality of the transformation (2 or 3), by default 3
        transform_type : str, optional
            Type of transformation ('rigid', 'similarity', or 'affine'), by default "affine"
        """
        self.dim = dim
        self.transform_type = transform_type.lower()
        self.dof = self._get_degrees_of_freedom()
        
    def _get_degrees_of_freedom(self) -> int:
        """Get number of degrees of freedom for the transformation."""
        if self.transform_type == "rigid":
            return 6 if self.dim == 3 else 3
        elif self.transform_type == "similarity":
            return 7 if self.dim == 3 else 4
        elif self.transform_type == "affine":
            return 12 if self.dim == 3 else 6
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")
    
    def parameters_to_matrix(
        self,
        parameters: torch.Tensor
    ) -> torch.Tensor:
        """Convert parameters to transformation matrix."""
        if self.transform_type == "rigid":
            return self._rigid_parameters_to_matrix(parameters)
        elif self.transform_type == "similarity":
            return self._similarity_parameters_to_matrix(parameters)
        else:  # affine
            return self._affine_parameters_to_matrix(parameters)
    
    def matrix_to_parameters(
        self,
        matrix: torch.Tensor
    ) -> torch.Tensor:
        """Convert transformation matrix to parameters."""
        if self.transform_type == "rigid":
            return self._matrix_to_rigid_parameters(matrix)
        elif self.transform_type == "similarity":
            return self._matrix_to_similarity_parameters(matrix)
        else:  # affine
            return self._matrix_to_affine_parameters(matrix)
            
    def _rigid_parameters_to_matrix(
        self,
        parameters: torch.Tensor
    ) -> torch.Tensor:
        """Convert rigid parameters (rotation + translation) to matrix."""
        device = parameters.device
        if self.dim == 3:
            rx, ry, rz, tx, ty, tz = parameters.split(1)
            
            # Create rotation matrices
            Rx = torch.tensor([[1, 0, 0],
                             [0, torch.cos(rx), -torch.sin(rx)],
                             [0, torch.sin(rx), torch.cos(rx)]], device=device)
            
            Ry = torch.tensor([[torch.cos(ry), 0, torch.sin(ry)],
                             [0, 1, 0],
                             [-torch.sin(ry), 0, torch.cos(ry)]], device=device)
            
            Rz = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0],
                             [torch.sin(rz), torch.cos(rz), 0],
                             [0, 0, 1]], device=device)
            
            R = Rz @ Ry @ Rx
            t = torch.stack([tx, ty, tz])
            
            # Create affine matrix
            matrix = torch.eye(4, device=device)
            matrix[:3, :3] = R
            matrix[:3, 3] = t.squeeze()
            
            return matrix
        else:  # 2D
            theta, tx, ty = parameters.split(1)
            c = torch.cos(theta)
            s = torch.sin(theta)
            
            matrix = torch.eye(3, device=device)
            matrix[:2, :2] = torch.tensor([[c, -s], [s, c]], device=device)
            matrix[:2, 2] = torch.stack([tx, ty]).squeeze()
            
            return matrix

class AffineRegistration:
    """
    Affine registration class supporting different transformation models.
    """
    
    def __init__(
        self,
        transform_type: str = "affine",
        iterations: int = 100,
        learning_rate: float = 0.1,
        metric: str = "ncc",
        optimizer_type: str = "adam",
        use_gpu: bool = True,
        verbose: bool = False,
        parameters: Optional[RegistrationParameters] = None
    ):
        """
        Initialize affine registration.

        Parameters
        ----------
        transform_type : str, optional
            Type of transformation ('rigid', 'similarity', or 'affine'), by default "affine"
        iterations : int, optional
            Number of optimization iterations, by default 100
        learning_rate : float, optional
            Learning rate for optimization, by default 0.1
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
        self.transform_type = transform_type
        
        # Initialize transformation
        self.transform = AffineTransform(transform_type=transform_type)
        
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
        initial_transform: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform affine registration.

        Parameters
        ----------
        fixed : Union[np.ndarray, str, Path]
            Fixed image or path to fixed image
        moving : Union[np.ndarray, str, Path]
            Moving image or path to moving image
        initial_transform : Optional[np.ndarray], optional
            Initial transformation matrix, by default None
        mask : Optional[np.ndarray], optional
            Mask for the fixed image, by default None

        Returns
        -------
        Dict[str, Any]
            Registration results including transformation matrix and warped image
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
            
        # Initialize parameters
        if initial_transform is None:
            parameters = torch.zeros(self.transform.dof, device=self.device)
        else:
            parameters = self.transform.matrix_to_parameters(
                torch.from_numpy(initial_transform).float().to(self.device)
            )
        parameters.requires_grad_(True)
        
        # Optimize
        result = self.optimizer.optimize(
            fixed_tensor,
            moving_tensor,
            lambda x, p: self._transform_image(x, p),
            lambda x, y: self._compute_loss(x, y, mask_tensor),
            parameters
        )
        
        # Get final transformation
        final_matrix = self.transform.parameters_to_matrix(result['final_parameters'])
        
        return {
            'transform_matrix': final_matrix.detach().cpu().numpy(),
            'warped_image': self._transform_image(
                moving_tensor,
                result['final_parameters']
            ).detach().cpu().numpy(),
            'parameters': result['final_parameters'].detach().cpu().numpy(),
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
        parameters: torch.Tensor
    ) -> torch.Tensor:
        """Apply transformation to image."""
        matrix = self.transform.parameters_to_matrix(parameters)
        grid = compute_grid(matrix, image.shape)
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
        return similarity