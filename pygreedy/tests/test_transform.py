"""
Tests for PyGreedy transform module.

Created by: devhliu
Created at: 2025-02-18 05:06:09 UTC
"""

import unittest
import torch
import numpy as np
from ..core.utils.transform import (
    compute_grid,
    compose_transforms,
    invert_transform,
    interpolate_deformation
)

class TestTransform(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shape = (64, 64)
        self.affine = torch.eye(4).to(self.device)
        self.displacement = torch.zeros(2, *self.shape).to(self.device)
        
    def test_compute_grid(self):
        """Test grid computation."""
        # Test affine grid
        grid = compute_grid(self.affine, self.shape)
        self.assertEqual(grid.shape, (*self.shape, 2))
        
        # Test displacement grid
        grid = compute_grid(self.displacement, self.shape)
        self.assertEqual(grid.shape, (*self.shape, 2))
        
    def test_compose_transforms(self):
        """Test transform composition."""
        # Test affine composition
        affine2 = self.affine.clone()
        composed = compose_transforms([self.affine, affine2], mode='affine')
        self.assertEqual(composed.shape, (4, 4))
        
        # Test displacement composition
        disp2 = self.displacement.clone()
        composed = compose_transforms(
            [self.displacement, disp2],
            mode='displacement'
        )
        self.assertEqual(composed.shape, self.displacement.shape)
        
    def test_invert_transform(self):
        """Test transform inversion."""
        # Test affine inversion
        inverse = invert_transform(self.affine, mode='affine')
        identity = torch.matmul(self.affine, inverse)
        self.assertTrue(
            torch.allclose(identity, torch.eye(4).to(self.device))
        )
        
        # Test displacement inversion
        inverse = invert_transform(self.displacement, mode='displacement')
        self.assertEqual(inverse.shape, self.displacement.shape)
        
    def test_interpolate_deformation(self):
        """Test deformation field interpolation."""
        # Test upsampling
        scale = 2.0
        interpolated = interpolate_deformation(self.displacement, scale)
        expected_shape = (
            self.displacement.shape[0],
            int(self.shape[0] * scale),
            int(self.shape[1] * scale)
        )
        self.assertEqual(interpolated.shape, expected_shape)
        
        # Test downsampling
        scale = 0.5
        interpolated = interpolate_deformation(self.displacement, scale)
        expected_shape = (
            self.displacement.shape[0],
            int(self.shape[0] * scale),
            int(self.shape[1] * scale)
        )
        self.assertEqual(interpolated.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()