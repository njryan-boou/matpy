---
layout: default
title: Home
---

# MatPy Documentation

[![Tests](https://github.com/njryan-boou/matpy/actions/workflows/tests.yml/badge.svg)](https://github.com/njryan-boou/matpy/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/matpy-linalg.svg)](https://badge.fury.io/py/matpy-linalg)
[![Python](https://img.shields.io/pypi/pyversions/matpy-linalg.svg)](https://pypi.org/project/matpy-linalg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library for vector and matrix operations with a clean, Pythonic API.

## Quick Links

- 📚 [API Reference](api-reference.md)
- 📖 [Tutorials](tutorials.md)
- 💡 [Examples](examples.md)
- 🔧 [Installation Guide](installation.md)
- 📦 [PyPI Package](https://pypi.org/project/matpy-linalg/)
- 🐙 [GitHub Repository](https://github.com/njryan-boou/matpy)

## Features

### Core Components

- **N-Dimensional Vectors** - Support for 2D, 3D, 4D, and beyond
- **Dynamic Matrices** - Any size with comprehensive operations
- **Linear System Solvers** - Gaussian elimination, LU, Cramer's rule, least squares
- **ODE Solvers** - Systems of differential equations
- **Coordinate Systems** - Cartesian, Polar, Spherical, Cylindrical conversions
- **Visualization** - 2D/3D plotting with matplotlib (optional)
- **Pure Python** - No required dependencies for core functionality

## Installation

```bash
# Core installation
pip install matpy-linalg

# With visualization support
pip install matpy-linalg[viz]
```

## Quick Start

```python
from matpy.vector.core import Vector
from matpy.matrix.core import Matrix

# Create vectors
v1 = Vector(3, 4, 5)
v2 = Vector(1, 2, 3)

# Vector operations
dot_product = v1.dot(v2)
magnitude = v1.magnitude()
unit_vector = v1.normalize()

# Create matrices
m1 = Matrix(2, 2, [[1, 2], [3, 4]])
m2 = Matrix(2, 2, [[5, 6], [7, 8]])

# Matrix operations
determinant = m1.determinant()
inverse = m1.inverse()
product = m1 @ m2  # Matrix multiplication
```

## Why MatPy?

- ✅ **Educational** - Clear, readable pure Python implementation
- ✅ **Comprehensive** - Full linear algebra toolkit in one package
- ✅ **Well-tested** - 300+ unit tests across multiple platforms
- ✅ **Type-safe** - Full type hints for better IDE support
- ✅ **Documented** - Extensive documentation and examples
- ✅ **Pythonic** - Clean API following Python conventions

## Support

- 📖 [Read the Docs](https://njryan-boou.github.io/matpy/)
- 🐛 [Report Issues](https://github.com/njryan-boou/matpy/issues)
- 💬 [Discussions](https://github.com/njryan-boou/matpy/discussions)

## License

MatPy is licensed under the MIT License. See [LICENSE](https://github.com/njryan-boou/matpy/blob/main/LICENSE) for details.
