"""
PyGreedy Optimizer Module
=======================

This module implements optimization strategies for image registration,
including multi-resolution optimization and various optimization algorithms.

Created by: devhliu
Created at: 2025-02-18 04:24:20 UTC
"""

import torch
import numpy as np
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    final_parameters: torch.Tensor
    convergence_achieved: bool
    final_loss: float
    iterations_per_level: List[int]
    loss_history: List[float]

class RegistrationOptimizer:
    """
    Optimizer for image registration supporting multi-resolution strategies
    and various optimization algorithms.
    """
    
    def __init__(
        self,
        optimizer_type: str = "adam",
        learning_rate: float = 0.1,
        num_levels: int = 3,
        scale_factor: float = 2.0,
        num_iterations: Optional[List[int]] = None,
        smoothing_sigmas: Optional[List[float]] = None,
        momentum: float = 0.9,
        convergence_eps: float = 1e-6,
        use_gpu: bool = True,
        verbose: bool = False
    ):
        """
        Initialize registration optimizer.

        Parameters
        ----------
        optimizer_type : str, optional
            Type of optimizer ('adam', 'sgd', or 'lbfgs'), by default "adam"
        learning_rate : float, optional
            Initial learning rate, by default 0.1
        num_levels : int, optional
            Number of multi-resolution levels, by default 3
        scale_factor : float, optional
            Scale factor between resolution levels, by default 2.0
        num_iterations : Optional[List[int]], optional
            List of iterations for each level, by default None
        smoothing_sigmas : Optional[List[float]], optional
            List of smoothing sigmas for each level, by default None
        momentum : float, optional
            Momentum factor for SGD, by default 0.9
        convergence_eps : float, optional
            Convergence criterion threshold, by default 1e-6
        use_gpu : bool, optional
            Whether to use GPU acceleration, by default True
        verbose : bool, optional
            Whether to print progress, by default False
        """
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        self.momentum = momentum
        self.convergence_eps = convergence_eps
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Set default iterations per level if not provided
        if num_iterations is None:
            self.num_iterations = [100 * (2 ** i) for i in range(num_levels)]
        else:
            self.num_iterations = num_iterations
            
        # Set default smoothing sigmas if not provided
        if smoothing_sigmas is None:
            self.smoothing_sigmas = [2.0 * (2 ** i) for i in range(num_levels)][::-1]
        else:
            self.smoothing_sigmas = smoothing_sigmas
            
    def optimize(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        transform_model: Callable,
        loss_function: Callable,
        initial_parameters: torch.Tensor
    ) -> OptimizationResult:
        """
        Perform multi-resolution optimization.

        Parameters
        ----------
        fixed : torch.Tensor
            Fixed image
        moving : torch.Tensor
            Moving image
        transform_model : Callable
            Function that applies transformation to moving image
        loss_function : Callable
            Function that computes similarity metric
        initial_parameters : torch.Tensor
            Initial transformation parameters

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        current_parameters = initial_parameters
        iterations_per_level = []
        loss_history = []
        
        for level in range(self.num_levels):
            if self.verbose:
                print(f"\nOptimization level {level + 1}/{self.num_levels}")
                
            # Resize images for current level
            scale_factor = self.scale_factor ** (self.num_levels - level - 1)
            fixed_scaled = self._resize_image(fixed, scale_factor)
            moving_scaled = self._resize_image(moving, scale_factor)
            
            # Scale parameters if needed
            if level > 0:
                current_parameters = self._scale_parameters(current_parameters, level)
                
            # Optimize at current level
            level_result = self._optimize_level(
                fixed_scaled,
                moving_scaled,
                transform_model,
                loss_function,
                current_parameters,
                level
            )
            
            current_parameters = level_result.final_parameters
            iterations_per_level.append(level_result.iterations)
            loss_history.extend(level_result.loss_history)
            
            if self.verbose:
                print(f"Level {level + 1} completed: {level_result.iterations} iterations")
                print(f"Final loss: {level_result.final_loss:.6f}")
                
        # Compute final loss at full resolution
        final_loss = self._compute_loss(
            fixed,
            moving,
            transform_model,
            loss_function,
            current_parameters
        )
        
        return OptimizationResult(
            final_parameters=current_parameters,
            convergence_achieved=True,
            final_loss=final_loss,
            iterations_per_level=iterations_per_level,
            loss_history=loss_history
        )
    
    def _create_optimizer(
        self,
        parameters: torch.Tensor,
        level: int
    ) -> torch.optim.Optimizer:
        """Create optimizer instance for current level."""
        lr = self.learning_rate * (self.scale_factor ** level)
        
        if self.optimizer_type == "adam":
            return torch.optim.Adam([parameters], lr=lr)
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD(
                [parameters],
                lr=lr,
                momentum=self.momentum
            )
        elif self.optimizer_type == "lbfgs":
            return torch.optim.LBFGS(
                [parameters],
                lr=lr,
                max_iter=20,
                line_search_fn="strong_wolfe"
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
            
    def _optimize_level(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        transform_model: Callable,
        loss_function: Callable,
        initial_parameters: torch.Tensor,
        level: int
    ) -> Dict[str, Any]:
        """Optimize registration at a single resolution level."""
        parameters = initial_parameters.clone().detach().requires_grad_(True)
        optimizer = self._create_optimizer(parameters, level)
        
        prev_loss = float('inf')
        loss_history = []
        num_iters = self.num_iterations[level]
        
        for i in range(num_iters):
            def closure():
                optimizer.zero_grad()
                transformed = transform_model(moving, parameters)
                loss = loss_function(fixed, transformed)
                loss.backward()
                return loss
                
            if self.optimizer_type == "lbfgs":
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
                
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.convergence_eps:
                if self.verbose:
                    print(f"Converged at iteration {i + 1}")
                break
                
            prev_loss = current_loss
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{num_iters}, Loss: {current_loss:.6f}")
                
        return {
            'final_parameters': parameters.detach(),
            'iterations': i + 1,
            'final_loss': current_loss,
            'loss_history': loss_history
        }
    
    @staticmethod
    def _resize_image(
        image: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """Resize image using interpolation."""
        if scale_factor == 1.0:
            return image
            
        return torch.nn.functional.interpolate(
            image.unsqueeze(0).unsqueeze(0),
            scale_factor=scale_factor,
            mode='trilinear' if image.dim() == 3 else 'bilinear',
            align_corners=True
        ).squeeze()
    
    @staticmethod
    def _scale_parameters(
        parameters: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """Scale parameters for current resolution level."""
        return parameters
    
    def _compute_loss(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        transform_model: Callable,
        loss_function: Callable,
        parameters: torch.Tensor
    ) -> float:
        """Compute loss at full resolution."""
        with torch.no_grad():
            transformed = transform_model(moving, parameters)
            loss = loss_function(fixed, transformed)
        return loss.item()