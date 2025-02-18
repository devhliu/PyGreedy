"""
Tests for PyGreedy CLI.

Created by: devhliu
Created at: 2025-02-18 05:06:09 UTC
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from ..cli.main import main

class TestCLI(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.fixed_path = self.test_dir / 'fixed.nii.gz'
        self.moving_path = self.test_dir / 'moving.nii.gz'
        self.output_dir = self.test_dir / 'output'
        
        # Create dummy test images
        self._create_test_images()
        
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)
        
    def test_register_command(self):
        """Test register command."""
        args = [
            'register',
            str(self.fixed_path),
            str(self.moving_path),
            '-o', str(self.output_dir),
            '--transform-type', 'affine',
            '--metric', 'ncc'
        ]
        
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        self.assertTrue((self.output_dir / 'warped.nii.gz').exists())
        
    def test_visualize_command(self):
        """Test visualize command."""
        output_path = self.test_dir / 'visualization.png'
        args = [
            'visualize',
            str(self.fixed_path),
            str(self.moving_path),
            '-o', str(output_path),
            '--type', 'overlay'
        ]
        
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        self.assertTrue(output_path.exists())
        
    def _create_test_images(self):
        """Create dummy test images."""
        import numpy as np
        import nibabel as nib
        
        # Create simple test images
        shape = (32, 32, 32)
        fixed = np.random.randn(*shape)
        moving = np.random.randn(*shape)
        
        # Save as NIfTI
        nib.save(nib.Nifti1Image(fixed, np.eye(4)), str(self.fixed_path))
        nib.save(nib.Nifti1Image(moving, np.eye(4)), str(self.moving_path))

if __name__ == '__main__':
    unittest.main()