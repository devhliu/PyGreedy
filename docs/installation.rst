Installation Guide
================

Created by: devhliu
Created at: 2025-02-18 05:27:54 UTC

Prerequisites
------------

Before installing PyGreedy, ensure you have:

- Python >= 3.8
- CUDA toolkit (optional, for GPU support)

Basic Installation
----------------

Install PyGreedy using pip:

.. code-block:: bash

   pip install pygreedy

Development Installation
----------------------

For development installation:

.. code-block:: bash

   git clone https://github.com/devhliu/pygreedy.git
   cd pygreedy
   pip install -e .[dev]

GPU Support
----------

For GPU acceleration, ensure you have:

1. NVIDIA GPU with CUDA support
2. CUDA toolkit installed
3. PyTorch with CUDA support

Verify Installation
-----------------

Verify your installation:

.. code-block:: python

   import pygreedy
   print(pygreedy.__version__)