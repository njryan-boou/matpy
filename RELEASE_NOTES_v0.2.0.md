# Release Notes - v0.2.0

## 🎉 Major Feature Release

MatPy v0.2.0 is a significant update that transforms the library from a basic 3D vector/matrix library into a comprehensive linear algebra toolkit.

## ✨ What's New

### N-Dimensional Vector Support
- **Breaking Change**: Vectors now support arbitrary dimensions (2D, 3D, 4D, and beyond)
- Changed from `Vector(x=0, y=0, z=0)` to `Vector(*args)`
- Maintains backward compatibility for 3D vectors

### Linear System Solvers
- Gaussian elimination with partial pivoting
- LU decomposition solver
- Cramer's rule implementation
- Least squares for overdetermined systems

### Differential Equation Solvers
- Homogeneous ODE systems (dx/dt = Ax)
- Non-homogeneous ODE systems (dx/dt = Ax + b(t))
- Matrix exponential computation
- Euler method and Runge-Kutta 4 numerical solvers

### Coordinate System Conversions
- 2D: Cartesian ↔ Polar ↔ Complex
- 3D: Cartesian ↔ Spherical ↔ Cylindrical
- Easy-to-use `VectorCoordinates` class

### Visualization Module (Optional)
- 2D and 3D vector plotting
- Vector field visualization
- Matrix heatmaps and grids
- Linear transformation visualization
- Coordinate system comparison plots
- Requires matplotlib (install with `pip install matpy-linalg[viz]`)

### Advanced Matrix Operations
- Hadamard product (element-wise multiplication)
- Kronecker product (tensor product)
- Row echelon form and RREF
- Matrix concatenation (horizontal/vertical)
- Additional matrix creation utilities

### Code Quality Improvements
- Centralized validation (`core/validate.py`)
- Centralized utilities (`core/utils.py`)
- 300+ unit tests with comprehensive coverage
- Improved error messages
- Better type hints

## 📦 Installation

```bash
# Core installation
pip install matpy-linalg

# With visualization support
pip install matpy-linalg[viz]
```

## 🔄 Migration Guide

### Breaking Changes

**Vector Constructor:**
```python
# Old (v0.1.0)
v = Vector(x=1, y=2, z=3)

# New (v0.2.0)
v = Vector(1, 2, 3)  # Works the same
v_2d = Vector(1, 2)  # Now possible!
v_4d = Vector(1, 2, 3, 4)  # Now possible!
```

**Module Imports:**
If you were importing from internal modules, some have moved:
```python
# Matrix operations now in separate module
from matpy.matrix import ops
from matpy.matrix import solve

# Vector operations now in separate module
from matpy.vector import ops
from matpy.vector.coordinates import VectorCoordinates
```

## 📊 Statistics

- **35 files** in the repository
- **9,000+ lines** of code
- **300+ unit tests**
- **5 major modules**: vector, matrix, core, visualization, error
- **40+ functions** and methods

## 🙏 Acknowledgments

Thank you to everyone who has shown interest in MatPy! This release represents a major step forward in creating a comprehensive, educational linear algebra library in pure Python.

## 📝 Full Changelog

### Added
- N-dimensional vector support
- Linear system solvers module (`matrix/solve.py`)
- Differential equation solvers
- Coordinate system conversions (`vector/coordinates.py`)
- Visualization module with matplotlib integration
- Matrix operations module (`matrix/ops.py`)
- Vector operations module (`vector/ops.py`)
- Centralized validation and utilities
- Hadamard and Kronecker products
- Row echelon form operations
- Matrix concatenation functions
- Comprehensive example files

### Changed
- Vector constructor now accepts `*args` instead of keyword arguments
- Reorganized module structure for better organization
- Improved error messages and validation
- Enhanced documentation and examples

### Fixed
- Various edge cases in matrix operations
- Numerical stability improvements
- Type hint corrections

---

**Full Documentation**: https://github.com/njryan-boou/matpy#readme
**PyPI Package**: https://pypi.org/project/matpy-linalg/
