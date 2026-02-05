# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-05

### Added
- **N-dimensional vector support** - Vectors now support arbitrary dimensions (2D, 3D, 4D+)
- **Linear system solvers module** (`matrix/solve.py`)
  - Gaussian elimination with partial pivoting
  - LU decomposition solver
  - Cramer's rule implementation
  - Least squares for overdetermined systems
- **Differential equation solvers**
  - Homogeneous ODE systems (dx/dt = Ax)
  - Non-homogeneous ODE systems (dx/dt = Ax + b(t))
  - Matrix exponential computation
  - Euler method numerical solver
  - Runge-Kutta 4 numerical solver
- **Coordinate system conversions** (`vector/coordinates.py`)
  - 2D: Cartesian ↔ Polar ↔ Complex
  - 3D: Cartesian ↔ Spherical ↔ Cylindrical
  - `VectorCoordinates` class with easy API
- **Visualization module** (optional, requires matplotlib)
  - 2D and 3D vector plotting (`plot_vector_2d`, `plot_vectors_2d`, `plot_vector_3d`, `plot_vectors_3d`)
  - Vector field visualization (`plot_vector_field_2d`)
  - Matrix heatmaps and grids (`plot_matrix_heatmap`, `plot_matrix_grid`)
  - Linear transformation visualization (`plot_transformation_2d`, `plot_transformation_3d`)
  - Coordinate system comparison plots (`plot_coordinate_systems_2d`, `plot_coordinate_systems_3d`)
- **Advanced matrix operations**
  - Hadamard product (element-wise multiplication)
  - Kronecker product (tensor product)
  - Row echelon form and RREF
  - Matrix concatenation (horizontal/vertical)
- **Matrix operations module** (`matrix/ops.py`)
  - Matrix creation utilities (`zeros`, `ones`, `identity`, `diagonal`, `from_rows`, `from_columns`)
  - Property tests (`is_diagonal`, `is_identity`, `is_upper_triangular`, `is_lower_triangular`)
- **Vector operations module** (`vector/ops.py`)
  - Advanced vector operations (projection, rejection, reflection, distance)
  - Geometric functions (angle calculation, component-wise min/max)
  - Interpolation (lerp, clamping)
  - Parallel and perpendicular tests
- **Core utilities**
  - Centralized validation (`core/validate.py`)
  - Centralized utility functions (`core/utils.py`)
- **Comprehensive examples**
  - `matrix_examples.py` - Complete matrix operations showcase
  - `visualization_examples.py` - Visualization demonstrations
- **GitHub Actions CI/CD**
  - Automated testing across multiple Python versions (3.7-3.12)
  - Multi-platform testing (Ubuntu, Windows, macOS)
  - Code coverage reporting
  - Automated PyPI publishing on releases
- **Project badges** in README
  - Build status
  - PyPI version
  - Python versions
  - License
  - Downloads
  - GitHub stars

### Changed
- **BREAKING**: Vector constructor now uses `Vector(*args)` instead of `Vector(x=0, y=0, z=0)`
  - `Vector(1, 2, 3)` - 3D vector (same as before)
  - `Vector(1, 2)` - 2D vector (now possible)
  - `Vector(1, 2, 3, 4)` - 4D vector (now possible)
- Reorganized module structure for better organization
- Improved error messages and validation throughout
- Enhanced documentation and examples
- Updated README with comprehensive feature documentation

### Fixed
- Various edge cases in matrix operations
- Numerical stability improvements in calculations
- Type hint corrections and improvements
- Import structure for better module organization

## [0.1.0] - 2025-12-20

### Added
- Initial release
- Basic 3D Vector class with operator support
- Matrix class with dynamic sizing
- Vector operations: dot product, cross product, magnitude, normalization
- Matrix operations: transpose, determinant, inverse, adjugate, trace, rank
- Custom error handling with descriptive exceptions
- Comprehensive test suite
- Pure Python implementation (no external dependencies)
- Examples and documentation

[0.2.0]: https://github.com/njryan-boou/matpy/releases/tag/v0.2.0
[0.1.0]: https://github.com/njryan-boou/matpy/releases/tag/v0.1.0
