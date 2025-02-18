"""
PyGreedy CLI Commands Module
=========================

Implementation of CLI commands for PyGreedy.

Created by: devhliu
Created at: 2025-02-18 05:02:23 UTC
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from ..core.parameters import RegistrationParameters
from ..core.affine import AffineRegistration
from ..core.diffeomorphic import DiffeomorphicRegistration
from ..core.utils.image_io import load_image, save_image
from ..core.utils.visualization import (
    plot_registration_result,
    plot_deformation_field,
    create_checkerboard,
    overlay_images
)

logger = logging.getLogger(__name__)

def register_command(args: argparse.Namespace) -> int:
    """
    Execute registration command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    int
        Exit code
    """
    try:
        # Load parameters
        params = load_parameters(args)
        
        # Load images
        fixed_img, fixed_affine = load_image(args.fixed)
        moving_img, moving_affine = load_image(args.moving)
        
        # Create registration object
        registration = create_registration(params)
        
        # Perform registration
        result = registration.register(
            fixed=fixed_img,
            moving=moving_img,
            mask=load_mask(args.mask) if args.mask else None
        )
        
        # Save results
        save_results(result, args)
        
        logger.info("Registration completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}", exc_info=True)
        return 1

def visualize_command(args: argparse.Namespace) -> int:
    """
    Execute visualization command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    int
        Exit code
    """
    try:
        # Load images
        fixed_img, _ = load_image(args.fixed)
        warped_img, _ = load_image(args.warped)
        
        # Create visualization based on type
        if args.type == 'checkerboard':
            result = create_checkerboard(fixed_img, warped_img)
        elif args.type == 'overlay':
            result = overlay_images(fixed_img, warped_img)
        elif args.type == 'difference':
            result = fixed_img - warped_img
            
        # Save or display result
        if args.output:
            save_image(args.output, result)
        else:
            plot_registration_result(
                fixed_img,
                warped_img,
                result,
                slice_idx=args.slice_index
            )
            
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}", exc_info=True)
        return 1

def load_parameters(args: argparse.Namespace) -> RegistrationParameters:
    """
    Load and configure registration parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    RegistrationParameters
        Registration parameters
    """
    if args.parameters:
        params = RegistrationParameters.load(args.parameters)
    else:
        params = RegistrationParameters()
        
    params.update({
        'transform_type': args.transform_type,
        'metric': args.metric,
        'num_levels': args.num_levels,
        'max_iterations': args.iterations,
        'learning_rate': args.learning_rate,
        'use_gpu': args.gpu,
        'verbose': args.verbose
    })
    
    return params

def create_registration(params: RegistrationParameters) -> Any:
    """
    Create registration object based on parameters.

    Parameters
    ----------
    params : RegistrationParameters
        Registration parameters

    Returns
    -------
    Any
        Registration object
    """
    if params.transform_type in ['rigid', 'affine']:
        return AffineRegistration(parameters=params)
    else:
        return DiffeomorphicRegistration(parameters=params)

def load_mask(path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load mask image if provided.

    Parameters
    ----------
    path : Optional[str]
        Path to mask image

    Returns
    -------
    Optional[np.ndarray]
        Mask image data or None
    """
    if path:
        mask_data, _ = load_image(path)
        return mask_data
    return None

def save_results(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Save registration results.

    Parameters
    ----------
    result : Dict[str, Any]
        Registration results
    args : argparse.Namespace
        Parsed command line arguments
    """
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save warped image
        save_image(
            output_dir / 'warped.nii.gz',
            result['warped_image'],
            result.get('warped_affine', None)
        )
        
        # Save transform
        if 'transform_matrix' in result:
            np.save(
                output_dir / 'transform.npy',
                result['transform_matrix']
            )
        
        if 'deformation_field' in result:
            np.save(
                output_dir / 'deformation.npy',
                result['deformation_field']
            )
            
            # Save deformation field visualization
            plot_deformation_field(
                result['deformation_field'],
                output_path=output_dir / 'deformation_field.png'
            )