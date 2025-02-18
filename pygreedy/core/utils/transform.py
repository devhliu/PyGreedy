"""
PyGreedy Transform Module
======================

This module provides utilities for handling spatial transformations,
including grid generation, composition, and interpolation.

Created by: devhliu
Created at: 2025-02-18 04:52:09 UTC
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, List
import logging

logger = logging.getLogger(__name__)

def compute_grid(
    transform: Union[torch.Tensor, np.ndarray],
    shape: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute sampling grid from transformation.

    Parameters
    ----------
    transform : Union[torch.Tensor, np.ndarray]
        Transformation matrix or displacement field
    shape : Optional[Tuple[int, ...]], optional
        Output shape, by default None (uses transform shape)
    device : Optional[torch.device], optional
        Device for computation, by default None

    Returns
    -------
    torch.Tensor
        Sampling grid suitable for grid_sample
    """
    # Convert numpy array to torch tensor if needed
    if isinstance(transform, np.ndarray):
        transform = torch.from_numpy(transform)
        
    if device is None:
        device = transform.device
        
    # Handle different types of transforms
    if transform.dim() == 2 and transform.shape == (4, 4):
        return _affine_grid(transform, shape, device)
    else:
        return _displacement_grid(transform, shape, device)

def compose_transforms(
    transforms: List[torch.Tensor],
    mode: str = 'affine'
) -> torch.Tensor:
    """
    Compose multiple transformations.

    Parameters
    ----------
    transforms : List[torch.Tensor]
        List of transformations to compose
    mode : str, optional
        Type of transformations ('affine' or 'displacement'), by default 'affine'

    Returns
    -------
    torch.Tensor
        Composed transformation
    """
    if not transforms:
        raise ValueError("Empty transform list")
        
    if mode == 'affine':
        return _compose_affine(transforms)
    elif mode == 'displacement':
        return _compose_displacement(transforms)
    else:
        raise ValueError(f"Unknown composition mode: {mode}")

def invert_transform(
    transform: torch.Tensor,
    mode: str = 'affine'
) -> torch.Tensor:
    """
    Compute inverse transformation.

    Parameters
    ----------
    transform : torch.Tensor
        Input transformation
    mode : str, optional
        Type of transformation ('affine' or 'displacement'), by default 'affine'

    Returns
    -------
    torch.Tensor
        Inverse transformation
    """
    if mode == 'affine':
        return torch.inverse(transform)
    elif mode == 'displacement':
        return _invert_displacement(transform)
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

def interpolate_deformation(
    deformation: torch.Tensor,
    scale_factor: Union[float, Tuple[float, ...]]
) -> torch.Tensor:
    """
    Interpolate deformation field to new size.

    Parameters
    ----------
    deformation : torch.Tensor
        Input deformation field
    scale_factor : Union[float, Tuple[float, ...]]
        Scale factor for interpolation

    Returns
    -------
    torch.Tensor
        Interpolated deformation field
    """
    if isinstance(scale_factor, (int, float)):
        scale_factor = [scale_factor] * (deformation.dim() - 1)
        
    # Reshape for interpolation
    shape = deformation.shape
    deformation = deformation.unsqueeze(0)
    
    # Interpolate
    interpolated = torch.nn.functional.interpolate(
        deformation,
        scale_factor=scale_factor,
        mode='trilinear' if deformation.dim() == 5 else 'bilinear',
        align_corners=True
    )
    
    return interpolated.squeeze(0)

def compute_jacobian(
    transform: torch.Tensor,
    mode: str = 'displacement'
) -> torch.Tensor:
    """
    Compute Jacobian determinant of transformation.

    Parameters
    ----------
    transform : torch.Tensor
        Input transformation
    mode : str, optional
        Type of transformation ('affine' or 'displacement'), by default 'displacement'

    Returns
    -------
    torch.Tensor
        Jacobian determinant
    """
    if mode == 'affine':
        return torch.det(transform[:3, :3])
    elif mode == 'displacement':
        return _compute_displacement_jacobian(transform)
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

def _affine_grid(
    matrix: torch.Tensor,
    shape: Optional[Tuple[int, ...]] = None,
    device: torch.device = None
) -> torch.Tensor:
    """Generate sampling grid from affine matrix."""
    if shape is None:
        raise ValueError("Shape must be provided for affine grid")
        
    # Create normalized coordinate grid
    if len(shape) == 2:
        grid = torch.meshgrid(
            torch.linspace(-1, 1, shape[0], device=device),
            torch.linspace(-1, 1, shape[1], device=device)
        )
    else:  # 3D
        grid = torch.meshgrid(
            torch.linspace(-1, 1, shape[0], device=device),
            torch.linspace(-1, 1, shape[1], device=device),
            torch.linspace(-1, 1, shape[2], device=device)
        )
        
    # Stack coordinates
    coords = torch.stack(grid).flatten(1)
    
    # Add homogeneous coordinate
    coords = torch.cat([coords, torch.ones_like(coords[:1])])
    
    # Transform coordinates
    transformed = torch.matmul(matrix, coords)
    
    # Normalize and reshape
    if len(shape) == 2:
        return transformed[:2].reshape(2, *shape).permute(1, 2, 0)
    else:
        return transformed[:3].reshape(3, *shape).permute(1, 2, 3, 0)

def _displacement_grid(
    displacement: torch.Tensor,
    shape: Optional[Tuple[int, ...]] = None,
    device: torch.device = None
) -> torch.Tensor:
    """Generate sampling grid from displacement field."""
    if shape is None:
        shape = displacement.shape[1:]
        
    # Create identity grid
    grid = _create_identity_grid(shape, device)
    
    # Add displacement
    return grid + displacement.permute(*range(1, len(shape) + 1), 0)

def _create_identity_grid(
    shape: Tuple[int, ...],
    device: torch.device = None
) -> torch.Tensor:
    """Create identity sampling grid."""
    if len(shape) == 2:
        grid = torch.meshgrid(
            torch.linspace(-1, 1, shape[0], device=device),
            torch.linspace(-1, 1, shape[1], device=device)
        )
    else:  # 3D
        grid = torch.meshgrid(
            torch.linspace(-1, 1, shape[0], device=device),
            torch.linspace(-1, 1, shape[1], device=device),
            torch.linspace(-1, 1, shape[2], device=device)
        )
        
    return torch.stack(grid).permute(*range(1, len(shape) + 1), 0)

def _compose_affine(
    transforms: List[torch.Tensor]
) -> torch.Tensor:
    """Compose affine transformations."""
    result = transforms[0]
    for transform in transforms[1:]:
        result = torch.matmul(transform, result)
    return result

def _compose_displacement(
    transforms: List[torch.Tensor]
) -> torch.Tensor:
    """Compose displacement fields."""
    result = transforms[0]
    for transform in transforms[1:]:
        grid = compute_grid(transform)
        result = torch.nn.functional.grid_sample(
            result.unsqueeze(0),
            grid.unsqueeze(0),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(0) + transform
    return result

def _invert_displacement(
    displacement: torch.Tensor,
    num_iterations: int = 10,
    learning_rate: float = 0.1
) -> torch.Tensor:
    """Compute inverse displacement field using fixed-point iteration."""
    inverse = torch.zeros_like(displacement)
    
    for _ in range(num_iterations):
        grid = compute_grid(inverse)
        composed = torch.nn.functional.grid_sample(
            displacement.unsqueeze(0),
            grid.unsqueeze(0),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(0)
        inverse = inverse - learning_rate * composed
        
    return inverse

def _compute_displacement_jacobian(
    displacement: torch.Tensor
) -> torch.Tensor:
    """Compute Jacobian determinant of displacement field."""
    gradients = torch.gradient(displacement)
    
    if len(gradients) == 2:  # 2D
        jac = torch.zeros(*displacement.shape[1:], 2, 2, device=displacement.device)
        jac[..., 0, 0] = 1 + gradients[0][0]
        jac[..., 0, 1] = gradients[1][0]
        jac[..., 1, 0] = gradients[0][1]
        jac[..., 1, 1] = 1 + gradients[1][1]
    else:  # 3D
        jac = torch.zeros(*displacement.shape[1:], 3, 3, device=displacement.device)
        jac[..., 0, 0] = 1 + gradients[0][0]
        jac[..., 0, 1] = gradients[1][0]
        jac[..., 0, 2] = gradients[2][0]
        jac[..., 1, 0] = gradients[0][1]
        jac[..., 1, 1] = 1 + gradients[1][1]
        jac[..., 1, 2] = gradients[2][1]
        jac[..., 2, 0] = gradients[0][2]
        jac[..., 2, 1] = gradients[1][2]
        jac[..., 2, 2] = 1 + gradients[2][2]
        
    return torch.det(jac)