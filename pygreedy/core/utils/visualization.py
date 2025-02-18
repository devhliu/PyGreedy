"""
PyGreedy Visualization Module
==========================

This module provides visualization utilities for registration results,
deformation fields, and optimization metrics.

Created by: devhliu
Created at: 2025-02-18 04:55:06 UTC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List, Union
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_registration_result(
    fixed: Union[np.ndarray, torch.Tensor],
    moving: Union[np.ndarray, torch.Tensor],
    warped: Union[np.ndarray, torch.Tensor],
    slice_idx: Optional[Union[int, Tuple[int, ...]]] = None,
    title: str = "Registration Result",
    output_path: Optional[Union[str, Path]] = None,
    fig_size: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot registration result comparison.

    Parameters
    ----------
    fixed : Union[np.ndarray, torch.Tensor]
        Fixed image
    moving : Union[np.ndarray, torch.Tensor]
        Moving image
    warped : Union[np.ndarray, torch.Tensor]
        Warped moving image
    slice_idx : Optional[Union[int, Tuple[int, ...]]], optional
        Slice indices for visualization, by default None
    title : str, optional
        Plot title, by default "Registration Result"
    output_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (15, 5)
    """
    # Convert to numpy if needed
    fixed = _to_numpy(fixed)
    moving = _to_numpy(moving)
    warped = _to_numpy(warped)
    
    # Select slice if not provided
    if slice_idx is None:
        slice_idx = tuple(s // 2 for s in fixed.shape)
    elif isinstance(slice_idx, int):
        slice_idx = (slice_idx,) * fixed.ndim
        
    # Extract slices
    fixed_slice = _extract_slice(fixed, slice_idx)
    moving_slice = _extract_slice(moving, slice_idx)
    warped_slice = _extract_slice(warped, slice_idx)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    fig.suptitle(title)
    
    # Plot images
    axes[0].imshow(fixed_slice, cmap='gray')
    axes[0].set_title('Fixed Image')
    axes[0].axis('off')
    
    axes[1].imshow(moving_slice, cmap='gray')
    axes[1].set_title('Moving Image')
    axes[1].axis('off')
    
    axes[2].imshow(warped_slice, cmap='gray')
    axes[2].set_title('Warped Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_deformation_field(
    deformation: Union[np.ndarray, torch.Tensor],
    slice_idx: Optional[Union[int, Tuple[int, ...]]] = None,
    spacing: int = 5,
    scale: float = 1.0,
    color: str = 'red',
    title: str = "Deformation Field",
    output_path: Optional[Union[str, Path]] = None,
    fig_size: Tuple[int, int] = (8, 8)
) -> None:
    """
    Plot deformation field vectors.

    Parameters
    ----------
    deformation : Union[np.ndarray, torch.Tensor]
        Deformation field
    slice_idx : Optional[Union[int, Tuple[int, ...]]], optional
        Slice indices for visualization, by default None
    spacing : int, optional
        Spacing between vectors, by default 5
    scale : float, optional
        Vector scale factor, by default 1.0
    color : str, optional
        Vector color, by default 'red'
    title : str, optional
        Plot title, by default "Deformation Field"
    output_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (8, 8)
    """
    # Convert to numpy if needed
    deformation = _to_numpy(deformation)
    
    # Select slice if not provided
    if slice_idx is None:
        slice_idx = tuple(s // 2 for s in deformation.shape[1:])
    elif isinstance(slice_idx, int):
        slice_idx = (slice_idx,) * (deformation.ndim - 1)
        
    # Extract slice
    def_slice = _extract_slice(deformation, slice_idx)
    
    # Create meshgrid
    y, x = np.mgrid[0:def_slice.shape[0]:spacing, 0:def_slice.shape[1]:spacing]
    u = def_slice[0, ::spacing, ::spacing]
    v = def_slice[1, ::spacing, ::spacing]
    
    # Create figure
    plt.figure(figsize=fig_size)
    plt.title(title)
    
    # Plot vectors
    plt.quiver(x, y, u, v, color=color, scale=scale, angles='xy')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_metric_history(
    metrics: Union[List[float], np.ndarray],
    title: str = "Optimization History",
    xlabel: str = "Iteration",
    ylabel: str = "Metric Value",
    output_path: Optional[Union[str, Path]] = None,
    fig_size: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot optimization metric history.

    Parameters
    ----------
    metrics : Union[List[float], np.ndarray]
        Metric values
    title : str, optional
        Plot title, by default "Optimization History"
    xlabel : str, optional
        X-axis label, by default "Iteration"
    ylabel : str, optional
        Y-axis label, by default "Metric Value"
    output_path : Optional[Union[str, Path]], optional
        Path to save figure, by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (10, 6)
    """
    plt.figure(figsize=fig_size)
    plt.plot(metrics)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def create_checkerboard(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    checker_size: int = 20
) -> np.ndarray:
    """
    Create checkerboard pattern from two images.

    Parameters
    ----------
    img1 : Union[np.ndarray, torch.Tensor]
        First image
    img2 : Union[np.ndarray, torch.Tensor]
        Second image
    checker_size : int, optional
        Size of checkerboard squares, by default 20

    Returns
    -------
    np.ndarray
        Checkerboard image
    """
    # Convert to numpy if needed
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
        
    # Create checkerboard mask
    x, y = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
    mask = ((x // checker_size) + (y // checker_size)) % 2
    
    # Apply mask
    result = img1.copy()
    result[mask == 1] = img2[mask == 1]
    
    return result

def overlay_images(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5,
    cmap1: str = 'Reds',
    cmap2: str = 'Blues'
) -> np.ndarray:
    """
    Create overlay of two images with different colormaps.

    Parameters
    ----------
    img1 : Union[np.ndarray, torch.Tensor]
        First image
    img2 : Union[np.ndarray, torch.Tensor]
        Second image
    alpha : float, optional
        Transparency level, by default 0.5
    cmap1 : str, optional
        Colormap for first image, by default 'Reds'
    cmap2 : str, optional
        Colormap for second image, by default 'Blues'

    Returns
    -------
    np.ndarray
        Overlay image
    """
    # Convert to numpy if needed
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
        
    # Normalize images
    img1 = _normalize_intensity(img1)
    img2 = _normalize_intensity(img2)
    
    # Create colormaps
    cmap1 = plt.get_cmap(cmap1)
    cmap2 = plt.get_cmap(cmap2)
    
    # Apply colormaps
    colored1 = cmap1(img1)
    colored2 = cmap2(img2)
    
    # Blend images
    overlay = alpha * colored1 + (1 - alpha) * colored2
    
    return overlay

def _to_numpy(
    array: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """Convert array to numpy if needed."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array

def _extract_slice(
    array: np.ndarray,
    indices: Tuple[int, ...]
) -> np.ndarray:
    """Extract 2D slice from n-dimensional array."""
    slices = tuple(slice(None) if i >= array.ndim else indices[i]
                  for i in range(array.ndim))
    return array[slices]

def _normalize_intensity(
    image: np.ndarray
) -> np.ndarray:
    """Normalize image intensities to [0, 1]."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)