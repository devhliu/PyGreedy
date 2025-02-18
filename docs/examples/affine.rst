Affine Registration Example
========================

Created by: devhliu
Created at: 2025-02-18 05:27:54 UTC

This example demonstrates how to perform affine registration using PyGreedy.

Basic Usage
----------

.. code-block:: python

   import pygreedy
   from pygreedy.core.parameters import RegistrationParameters

   # Configure parameters
   params = RegistrationParameters(
       transform_type='affine',
       metric='ncc',
       num_levels=3,
       max_iterations=100
   )

   # Load images
   fixed_image = pygreedy.load_image('fixed.nii.gz')
   moving_image = pygreedy.load_image('moving.nii.gz')

   # Perform registration
   result = pygreedy.register_images(fixed_image, moving_image, params)

   # Save results
   pygreedy.save_image('warped.nii.gz', result['warped_image'])
   np.save('transform.npy', result['transform_matrix'])

Advanced Usage
-------------

.. code-block:: python

   # Configure advanced parameters
   params = RegistrationParameters(
       transform_type='affine',
       metric='mi',
       num_levels=4,
       max_iterations=200,
       learning_rate=0.1,
       regularization_weight=0.1,
       use_gpu=True
   )

   # Add mask
   mask = pygreedy.load_image('mask.nii.gz')
   result = pygreedy.register_images(
       fixed_image,
       moving_image,
       params,
       mask=mask
   )

   # Visualize results
   pygreedy.visualize.plot_registration_result(
       fixed_image,
       moving_image,
       result['warped_image'],
       title='Affine Registration Result'
   )