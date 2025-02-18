# PyGreedy

A flexible medical image registration package using greedy algorithms.

Created by: devhliu
Created at: 2025-02-18 05:22:28 UTC

## Overview

PyGreedy is a Python package for medical image registration that implements greedy optimization strategies. It supports both affine and diffeomorphic transformations, multiple similarity metrics, and provides both a command-line interface and a Python API.

## Features

- **Registration Methods**
  - Affine registration (rigid, similarity, affine)
  - Diffeomorphic registration
  - Multi-resolution optimization

- **Similarity Metrics**
  - Normalized Cross Correlation (NCC)
  - Mutual Information (MI)
  - Mean Squared Error (MSE)

- **Key Features**
  - GPU acceleration with PyTorch
  - DICOM and NIfTI support
  - Transform composition and inversion
  - Registration visualization tools
  - Comprehensive logging

## Installation

```bash
pip install pygreedy
```

For development installation:

```bash
git clone https://github.com/devhliu/pygreedy.git
cd pygreedy
pip install -e .[dev]
```

## Quick Start

### Command Line Interface

1. **Basic Registration**
```bash
pygreedy register fixed.nii.gz moving.nii.gz -o output_dir
```

2. **Advanced Registration**
```bash
pygreedy register fixed.nii.gz moving.nii.gz \
    --transform-type diffeomorphic \
    --metric ncc \
    --num-levels 4 \
    --iterations 200 \
    --gpu \
    -o output_dir
```

3. **Visualization**
```bash
pygreedy visualize fixed.nii.gz warped.nii.gz \
    --type overlay \
    --output result.png
```

### Python API

```python
from pygreedy import register_images
from pygreedy.core.parameters import RegistrationParameters

# Configure registration
params = RegistrationParameters(
    transform_type='affine',
    metric='ncc',
    num_levels=3,
    max_iterations=100,
    use_gpu=True
)

# Perform registration
result = register_images(fixed_image, moving_image, parameters=params)

# Access results
warped_image = result['warped_image']
transform = result['transform_matrix']  # for affine
deformation = result['deformation_field']  # for diffeomorphic
```

## Documentation

Detailed documentation is available at [Read the Docs](https://pygreedy.readthedocs.io/).

### Building Documentation

```bash
cd docs
make html
```

Documentation will be available at `docs/build/html/index.html`.

## Testing

Run the test suite:

```bash
python -m unittest discover pygreedy/tests
```

Run with coverage:

```bash
coverage run -m unittest discover pygreedy/tests
coverage report
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

### Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install development dependencies:
```bash
pip install -e .[dev]
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyGreedy in your research, please cite:

```bibtex
@software{pygreedy2025,
  author = {Liu, Dev H.},
  title = {PyGreedy: A Flexible Medical Image Registration Package},
  year = {2025},
  url = {https://github.com/devhliu/pygreedy}
}
```

## Acknowledgments

- Thanks to the PyTorch team for their excellent deep learning framework
- Thanks to the medical imaging community for their valuable feedback and contributions

## Contact

devhliu - huiliu.liu@gmail.com

Project Link: [https://github.com/devhliu/pygreedy](https://github.com/devhliu/pygreedy)