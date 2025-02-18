"""
PyGreedy CLI Module
=================

Command-line interface for PyGreedy image registration package.

Created by: devhliu
Created at: 2025-02-18 05:02:23 UTC
"""

from .main import main
from .commands import register_command, visualize_command

__all__ = ['main', 'register_command', 'visualize_command']