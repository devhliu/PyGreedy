"""
PyGreedy CLI Parser Module
========================

Command-line argument parser for PyGreedy.

Created by: devhliu
Created at: 2025-02-18 05:02:23 UTC
"""

import argparse
from typing import Optional

def create_parser() -> argparse.ArgumentParser:
    """
    Create main argument parser for PyGreedy CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="PyGreedy: Medical Image Registration Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title='commands',
        dest='command'
    )
    
    # Add register command parser
    register_parser = subparsers.add_parser(
        'register',
        help='Register two images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_register_arguments(register_parser)
    
    # Add visualize command parser
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Visualize registration results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_visualize_arguments(visualize_parser)
    
    return parser

def _add_register_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for register command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to
    """
    # Required arguments
    parser.add_argument(
        'fixed',
        type=str,
        help='Path to fixed image'
    )
    parser.add_argument(
        'moving',
        type=str,
        help='Path to moving image'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '--parameters',
        '-p',
        type=str,
        help='Path to parameters JSON file'
    )
    parser.add_argument(
        '--mask',
        '-m',
        type=str,
        help='Path to mask image'
    )
    parser.add_argument(
        '--transform-type',
        '-t',
        choices=['rigid', 'affine', 'diffeomorphic'],
        default='affine',
        help='Type of registration transform'
    )
    parser.add_argument(
        '--metric',
        choices=['ncc', 'mi', 'mse'],
        default='ncc',
        help='Similarity metric'
    )
    parser.add_argument(
        '--num-levels',
        type=int,
        default=3,
        help='Number of multi-resolution levels'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Maximum number of iterations'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration'
    )

def _add_visualize_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for visualize command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to
    """
    parser.add_argument(
        'fixed',
        type=str,
        help='Path to fixed image'
    )
    parser.add_argument(
        'warped',
        type=str,
        help='Path to warped image'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file path'
    )
    parser.add_argument(
        '--type',
        '-t',
        choices=['checkerboard', 'overlay', 'difference'],
        default='overlay',
        help='Type of visualization'
    )
    parser.add_argument(
        '--slice-index',
        '-s',
        type=int,
        help='Slice index for visualization'
    )