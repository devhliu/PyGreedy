"""
PyGreedy Parameters Module
========================

This module defines the parameter management system for the registration framework,
including default values, validation, and parameter grouping.

Created by: devhliu
Created at: 2025-02-18 04:32:40 UTC
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import json
from pathlib import Path

@dataclass
class RegistrationParameters:
    """
    Container for registration parameters with validation and grouping.
    
    This class manages all parameters used in the registration process,
    providing default values, validation, and parameter grouping functionality.
    """
    
    # Multi-resolution parameters
    num_levels: int = 3
    scale_factor: float = 2.0
    
    # Optimization parameters
    max_iterations: int = 200
    learning_rate: float = 0.1
    convergence_eps: float = 1e-6
    optimizer_type: str = "adam"
    momentum: float = 0.9
    
    # Transform parameters
    transform_type: str = "diffeomorphic"
    scaling_steps: int = 6
    regularization_weight: float = 0.1
    
    # Similarity metric parameters
    metric: str = "ncc"
    metric_params: Dict[str, Any] = field(default_factory=lambda: {
        "ncc": {"local_window": None, "eps": 1e-8},
        "mi": {"num_bins": 32, "sigma": 0.1},
        "mse": {},
        "ngf": {"eps": 1e-8}
    })
    
    # Interpolation parameters
    interpolation_mode: str = "bilinear"
    padding_mode: str = "zeros"
    align_corners: bool = True
    
    # GPU parameters
    use_gpu: bool = True
    gpu_id: int = 0
    
    # Output parameters
    save_intermediate: bool = False
    output_dir: Optional[str] = None
    verbose: bool = False
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate parameter values.
        
        Raises
        ------
        ValueError
            If any parameter has an invalid value
        """
        self._validate_multi_resolution()
        self._validate_optimization()
        self._validate_transform()
        self._validate_metric()
        self._validate_interpolation()
        self._validate_gpu()
    
    def _validate_multi_resolution(self) -> None:
        """Validate multi-resolution parameters."""
        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        if self.scale_factor <= 1.0:
            raise ValueError("scale_factor must be > 1.0")
    
    def _validate_optimization(self) -> None:
        """Validate optimization parameters."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.convergence_eps <= 0:
            raise ValueError("convergence_eps must be > 0")
        if self.optimizer_type not in ["adam", "sgd", "lbfgs"]:
            raise ValueError("Invalid optimizer_type")
        if not 0 <= self.momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
    
    def _validate_transform(self) -> None:
        """Validate transform parameters."""
        if self.transform_type not in ["rigid", "affine", "diffeomorphic"]:
            raise ValueError("Invalid transform_type")
        if self.scaling_steps < 1:
            raise ValueError("scaling_steps must be >= 1")
        if self.regularization_weight < 0:
            raise ValueError("regularization_weight must be >= 0")
    
    def _validate_metric(self) -> None:
        """Validate metric parameters."""
        if self.metric not in ["ncc", "mi", "mse", "ngf"]:
            raise ValueError("Invalid metric")
    
    def _validate_interpolation(self) -> None:
        """Validate interpolation parameters."""
        if self.interpolation_mode not in ["nearest", "bilinear", "bicubic"]:
            raise ValueError("Invalid interpolation_mode")
        if self.padding_mode not in ["zeros", "border", "reflection"]:
            raise ValueError("Invalid padding_mode")
    
    def _validate_gpu(self) -> None:
        """Validate GPU parameters."""
        if self.gpu_id < 0:
            raise ValueError("gpu_id must be >= 0")
    
    def update(
        self,
        params: Dict[str, Any]
    ) -> None:
        """
        Update parameters with new values.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to update
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of parameters
        """
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }
    
    def save(
        self,
        filename: Union[str, Path]
    ) -> None:
        """
        Save parameters to JSON file.

        Parameters
        ----------
        filename : Union[str, Path]
            Path to output file
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(
        cls,
        filename: Union[str, Path]
    ) -> 'RegistrationParameters':
        """
        Load parameters from JSON file.

        Parameters
        ----------
        filename : Union[str, Path]
            Path to input file

        Returns
        -------
        RegistrationParameters
            Loaded parameters
        """
        with open(filename, 'r') as f:
            params = json.load(f)
        return cls(**params)
    
    def get_metric_params(
        self,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get parameters for specific metric.

        Parameters
        ----------
        metric_name : Optional[str], optional
            Name of metric, by default None (uses self.metric)

        Returns
        -------
        Dict[str, Any]
            Metric parameters
        """
        metric_name = metric_name or self.metric
        return self.metric_params[metric_name].copy()
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """
        Get optimizer parameters.

        Returns
        -------
        Dict[str, Any]
            Optimizer parameters
        """
        return {
            'type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'convergence_eps': self.convergence_eps
        }
    
    def get_transform_params(self) -> Dict[str, Any]:
        """
        Get transform parameters.

        Returns
        -------
        Dict[str, Any]
            Transform parameters
        """
        return {
            'type': self.transform_type,
            'scaling_steps': self.scaling_steps,
            'regularization_weight': self.regularization_weight
        }