"""
Tests for PyGreedy metrics module.

Created by: devhliu
Created at: 2025-02-18 05:06:09 UTC
"""

import unittest
import torch
import numpy as np
from ..core.metrics import (
    NormalizedCrossCorrelation,
    MutualInformation,
    MeanSquaredError
)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fixed = torch.randn(128, 128).to(self.device)
        self.moving = torch.randn(128, 128).to(self.device)
        self.mask = torch.ones_like(self.fixed).to(self.device)
        
    def test_ncc(self):
        """Test NCC metric."""
        ncc = NormalizedCrossCorrelation()
        value = ncc(self.fixed, self.moving)
        
        self.assertIsInstance(value, torch.Tensor)
        self.assertTrue(0 <= value <= 2)  # NCC range is [-1, 1], dissimilarity is [0, 2]
        
        # Test with mask
        value_masked = ncc(self.fixed, self.moving, self.mask)
        self.assertIsInstance(value_masked, torch.Tensor)
        
    def test_mi(self):
        """Test MI metric."""
        mi = MutualInformation()
        value = mi(self.fixed, self.moving)
        
        self.assertIsInstance(value, torch.Tensor)
        self.assertTrue(value >= 0)  # MI is non-negative
        
        # Test with different number of bins
        mi_32 = MutualInformation(num_bins=32)
        value_32 = mi_32(self.fixed, self.moving)
        self.assertIsInstance(value_32, torch.Tensor)
        
    def test_mse(self):
        """Test MSE metric."""
        mse = MeanSquaredError()
        value = mse(self.fixed, self.moving)
        
        self.assertIsInstance(value, torch.Tensor)
        self.assertTrue(value >= 0)  # MSE is non-negative
        
        # Test with identical images
        value_identical = mse(self.fixed, self.fixed)
        self.assertAlmostEqual(value_identical.item(), 0, places=5)

if __name__ == '__main__':
    unittest.main()