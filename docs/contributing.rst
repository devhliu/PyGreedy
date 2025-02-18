Contributing Guide
===============

Created by: devhliu
Created at: 2025-02-18 05:27:54 UTC

Thank you for considering contributing to PyGreedy! This document provides guidelines and instructions for contribution.

Development Setup
---------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/yourusername/pygreedy.git
      cd pygreedy

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/Mac
      venv\Scripts\activate     # Windows

4. Install development dependencies:

   .. code-block:: bash

      pip install -e .[dev]

5. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Code Style
---------

- Follow PEP 8
- Use type hints
- Write docstrings (NumPy style)
- Keep line length to 88 characters
- Use Black for formatting

Testing
-------

1. Run tests:

   .. code-block:: bash

      python -m unittest discover pygreedy/tests

2. Run with coverage:

   .. code-block:: bash

      coverage run -m unittest discover pygreedy/tests
      coverage report

Documentation
------------

1. Install documentation dependencies:

   .. code-block:: bash

      pip install -e .[docs]

2. Build documentation:

   .. code-block:: bash

      cd docs
      make html

Pull Request Process
------------------

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Run tests and linting
6. Submit PR

Code Review
----------

- All PRs require review
- Address review comments
- Keep PRs focused
- Update PR based on feedback