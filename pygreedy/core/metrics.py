"""
PyGreedy Metrics Module
=====================

This module implements various similarity metrics for image registration,
including normalized cross-correlation, mutual information, and mean squared error.

Created by: devhliu
Created at: 2025-02-18 04:26:49 UTC
"""

import torch
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod

class SimilarityMetric(ABC):
    """Abstract base class for similarity metrics."""
    
    def __init__(self):
        """Initialize metric."""
        pass
    
    @abstractmethod
    def __call__(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute similarity metric between fixed and moving images.

        Parameters
        ----------
        fixed : torch.Tensor
            Fixed image tensor
        moving : torch.Tensor
            Moving image tensor
        mask : Optional[torch.Tensor], optional
            Mask tensor for masked computation, by default None

        Returns
        -------
        torch.Tensor
            Similarity metric value
        """
        pass

class NormalizedCrossCorrelation(SimilarityMetric):
    """
    Normalized Cross-Correlation (NCC) similarity metric.
    
    NCC is invariant to linear intensity transformations and is suitable
    for mono-modal registration.
    """
    
    def __init__(
        self,
        eps: float = 1e-8,
        local_window: Optional[int] = None
    ):
        """
        Initialize NCC metric.

        Parameters
        ----------
        eps : float, optional
            Small constant for numerical stability, by default 1e-8
        local_window : Optional[int], optional
            Size of local window for local NCC, by default None
        """
        super().__init__()
        self.eps = eps
        self.local_window = local_window
        
    def __call__(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute NCC between fixed and moving images."""
        if self.local_window is not None:
            return self._local_ncc(fixed, moving, mask)
        
        # Center the images by subtracting their means
        f_mean = torch.mean(fixed)
        m_mean = torch.mean(moving)
        f_centered = fixed - f_mean
        m_centered = moving - m_mean
        
        # Compute correlation
        numerator = torch.sum(f_centered * m_centered)
        denominator = torch.sqrt(
            torch.sum(f_centered ** 2) * torch.sum(m_centered ** 2) + self.eps
        )
        
        ncc = numerator / denominator
        
        # Convert to dissimilarity measure (1 - NCC)
        return 1.0 - ncc
    
    def _local_ncc(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute local NCC using sliding window."""
        # Implement local NCC with sliding window
        # TODO: Implement efficient local NCC computation
        raise NotImplementedError("Local NCC not yet implemented")

class MutualInformation(SimilarityMetric):
    """
    Mutual Information (MI) similarity metric.
    
    MI is suitable for multi-modal registration as it measures statistical
    dependency between images without assuming a linear relationship.
    """
    
    def __init__(
        self,
        num_bins: int = 32,
        sigma: float = 0.1
    ):
        """
        Initialize MI metric.

        Parameters
        ----------
        num_bins : int, optional
            Number of histogram bins, by default 32
        sigma : float, optional
            Parzen window width, by default 0.1
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        
    def __call__(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute MI between fixed and moving images."""
        # Normalize images to [0, 1]
        fixed = self._normalize_intensity(fixed)
        moving = self._normalize_intensity(moving)
        
        # Compute joint histogram
        joint_hist = self._compute_joint_histogram(fixed, moving)
        
        # Compute marginal histograms
        p_fixed = torch.sum(joint_hist, dim=1)
        p_moving = torch.sum(joint_hist, dim=0)
        
        # Compute entropies
        h_fixed = self._compute_entropy(p_fixed)
        h_moving = self._compute_entropy(p_moving)
        h_joint = self._compute_entropy(joint_hist.flatten())
        
        # Compute mutual information
        mi = h_fixed + h_moving - h_joint
        
        # Convert to dissimilarity measure
        return -mi
    
    def _normalize_intensity(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """Normalize image intensities to [0, 1]."""
        min_val = torch.min(image)
        max_val = torch.max(image)
        return (image - min_val) / (max_val - min_val + 1e-8)
    
    def _compute_joint_histogram(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint histogram using Parzen window estimation."""
        # Create bin centers
        bin_centers = torch.linspace(0, 1, self.num_bins, device=fixed.device)
        
        # Reshape images and create meshgrid
        f_flat = fixed.flatten().unsqueeze(1)
        m_flat = moving.flatten().unsqueeze(1)
        
        # Compute Parzen window contributions
        dist_f = torch.exp(-(f_flat - bin_centers) ** 2 / (2 * self.sigma ** 2))
        dist_m = torch.exp(-(m_flat - bin_centers) ** 2 / (2 * self.sigma ** 2))
        
        # Normalize distributions
        dist_f = dist_f / (torch.sum(dist_f, dim=1, keepdim=True) + 1e-8)
        dist_m = dist_m / (torch.sum(dist_m, dim=1, keepdim=True) + 1e-8)
        
        # Compute joint histogram
        joint_hist = torch.mm(dist_f.t(), dist_m)
        joint_hist = joint_hist / (torch.sum(joint_hist) + 1e-8)
        
        return joint_hist
    
    def _compute_entropy(
        self,
        p: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy from probability distribution."""
        return -torch.sum(p * torch.log2(p + 1e-8))

class MeanSquaredError(SimilarityMetric):
    """
    Mean Squared Error (MSE) similarity metric.
    
    MSE is suitable for mono-modal registration when intensities are
    directly comparable.
    """
    
    def __call__(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute MSE between fixed and moving images."""
        squared_diff = (fixed - moving) ** 2
        
        if mask is not None:
            squared_diff = squared_diff * mask
            return torch.sum(squared_diff) / (torch.sum(mask) + 1e-8)
        
        return torch.mean(squared_diff)

class NormalizedGradientFieldMetric(SimilarityMetric):
    """
    Normalized Gradient Field (NGF) similarity metric.
    
    NGF is suitable for multi-modal registration as it compares
    image structure through gradients rather than intensities.
    """
    
    def __init__(
        self,
        eps: float = 1e-8
    ):
        """
        Initialize NGF metric.

        Parameters
        ----------
        eps : float, optional
            Small constant for numerical stability, by default 1e-8
        """
        super().__init__()
        self.eps = eps
        
    def __call__(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute NGF metric between fixed and moving images."""
        # Compute gradients
        f_grad = self._compute_gradients(fixed)
        m_grad = self._compute_gradients(moving)
        
        # Normalize gradients
        f_grad_norm = self._normalize_gradients(f_grad)
        m_grad_norm = self._normalize_gradients(m_grad)
        
        # Compute NGF metric
        similarity = torch.sum(
            torch.sum(f_grad_norm * m_grad_norm, dim=0)
        )
        
        if mask is not None:
            similarity = similarity * mask
            
        return 1.0 - similarity / (torch.numel(fixed) + self.eps)
    
    def _compute_gradients(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """Compute image gradients using central differences."""
        gradients = torch.gradient(image)
        return torch.stack(gradients)
    
    def _normalize_gradients(
        self,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """Normalize gradients."""
        grad_norm = torch.sqrt(
            torch.sum(gradients ** 2, dim=0) + self.eps
        )
        return gradients / grad_norm.unsqueeze(0)