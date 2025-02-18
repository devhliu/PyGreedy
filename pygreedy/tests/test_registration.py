"""
Tests for PyGreedy registration module.

Created by: devhliu
Created at: 2025-02-18 05:06:09 UTC
"""

import unittest
import torch
import numpy as np
from ..core.parameters import RegistrationParameters
from ..core.affine import AffineRegistration
from ..core.diffeomorphic import DiffeomorphicRegistration

class TestRegistration(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shape = (64, 64)
        self.fixed = torch.randn(*self.shape).to(self.device)
        self.moving = torch.randn(*self.shape).to(self.device)
        self.params = RegistrationParameters()
        
    def test_affine_registration(self):
        """Test affine registration."""
        registration = AffineRegistration(parameters=self.params)
        result = registration.register(self.fixed, self.moving)
        
        self.assertIn('warped_image', result)
        self.assertIn('transform_matrix', result)
        self.assertEqual(result['warped_image'].shape, self.shape)
        self.assertEqual(result['transform_matrix'].shape, (4, 4))
        
    def test_diffeomorphic_registration(self):
        """Test diffeomorphic registration."""
        self.params.transform_type = 'diffeomorphic'
        registration = DiffeomorphicRegistration(parameters=self.params)
        result = registration.register(self.fixed, self.moving)
        
        self.assertIn('warped_image', result)
        self.assertIn('deformation_field', result)
        self.assertEqual(result['warped_image'].shape, self.shape)
        self.assertEqual(
            result['deformation_field'].shape,
            (2, *self.shape)
        )
        
    def test_registration_with_mask(self):
        """Test registration with mask."""
        mask = torch.ones_like(self.fixed).to(self.device)
        
        # Test affine registration
        registration = AffineRegistration(parameters=self.params)
        result = registration.register(self.fixed, self.moving, mask=mask)
        self.assertIn('warped_image', result)
        
        # Test diffeomorphic registration
        self.params.transform_type = 'diffeomorphic'
        registration = DiffeomorphicRegistration(parameters=self.params)
        result = registration.register(self.fixed, self.moving, mask=mask)
        self.assertIn('warped_image', result)

if __name__ == '__main__':
    unittest.main()