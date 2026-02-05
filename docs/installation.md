---
layout: default
title: Installation Guide
---

# Installation Guide

[← Back to Home](index.md)

## Requirements

- Python 3.7 or higher
- pip (Python package installer)

## Installation Methods

### From PyPI (Recommended)

The easiest way to install MatPy is from PyPI:

```bash
# Core installation
pip install matpy-linalg

# With visualization support
pip install matpy-linalg[viz]

# With all development tools
pip install matpy-linalg[dev]
```

### From Source

Clone the repository and install in development mode:

```bash
# Clone the repository
git clone https://github.com/njryan-boou/matpy.git
cd matpy

# Install in development mode
pip install -e .

# Or install with visualization support
pip install -e ".[viz]"

# Or install with all development tools
pip install -e ".[dev]"
```

### From GitHub Release

Download the latest wheel or source distribution from the [releases page](https://github.com/njryan-boou/matpy/releases):

```bash
# Install from wheel
pip install matpy_linalg-0.2.0-py3-none-any.whl

# Or install from source
pip install matpy_linalg-0.2.0.tar.gz
```

## Optional Dependencies

### Visualization Support

For plotting and visualization features:

```bash
pip install matpy-linalg[viz]
```

This installs:
- matplotlib >= 3.5

### Development Tools

For contributors and developers:

```bash
pip install matpy-linalg[dev]
```

This installs:
- pytest >= 7.0 (testing)
- pytest-cov >= 4.0 (coverage)
- black >= 23.0 (formatting)
- flake8 >= 6.0 (linting)
- mypy >= 1.0 (type checking)

## Verify Installation

After installation, verify it works:

```python
# Test basic imports
from matpy import Vector, Matrix
print("MatPy installed successfully!")

# Test vector creation
v = Vector(1, 2, 3)
print(f"Vector: {v}")

# Test matrix creation
m = Matrix(2, 2, [[1, 2], [3, 4]])
print(f"Matrix determinant: {m.determinant()}")
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade matpy-linalg
```

## Uninstallation

To remove MatPy:

```bash
pip uninstall matpy-linalg
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure MatPy is installed in the correct Python environment:

```bash
python -m pip list | grep matpy
```

### Visualization Issues

If visualization features don't work, ensure matplotlib is installed:

```bash
pip install matplotlib>=3.5
```

Or reinstall with visualization support:

```bash
pip install --force-reinstall matpy-linalg[viz]
```

### Platform-Specific Issues

**Windows:**
- Use Command Prompt or PowerShell
- May need to use `python` instead of `python3`

**macOS/Linux:**
- May need to use `pip3` instead of `pip`
- May need `sudo` for system-wide installation (not recommended, use virtual environments)

## Virtual Environments

We recommend using virtual environments:

```bash
# Create virtual environment
python -m venv matpy-env

# Activate (Windows)
matpy-env\Scripts\activate

# Activate (macOS/Linux)
source matpy-env/bin/activate

# Install MatPy
pip install matpy-linalg
```

## Next Steps

- [Tutorials](tutorials.md) - Learn how to use MatPy
- [API Reference](api-reference.md) - Detailed API documentation
- [Examples](examples.md) - Code examples

[← Back to Home](index.md)
