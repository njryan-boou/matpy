# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-08

### Added
- **Complex eigenvalue support** - Eigenvalues and eigenvectors now handle complex numbers
  - 2x2 matrices return complex conjugate pairs when discriminant < 0
  - 3x3 matrices return all eigenvalues including complex conjugates
  - Eigenvectors normalized correctly for complex numbers using `abs()` for magnitude
- **New Matrix instance methods** - Cleaner API with property checks as instance methods
  - `is_diagonal()` - Check if all non-diagonal elements are zero
  - `is_identity()` - Check if matrix is the identity matrix
  - `is_upper_triangular()` - Check if all elements below diagonal are zero
  - `is_lower_triangular()` - Check if all elements above diagonal are zero
- **Memory optimization** - `__slots__` added to Vector and Matrix classes
  - ~40% memory reduction per instance
  - Faster attribute access
  - Prevents accidental attribute creation

### Changed
- **Code reorganization** - Cleaner separation between instance methods and factory functions
  - `core.py` - Single-matrix operations as instance/static methods
  - `ops.py` - Factory functions and multi-matrix operations only
  - Removed ~170 lines of duplicate wrapper functions
- **ODE solver improvements** - Enhanced differential equation solver
  - Now handles complex eigenvalues correctly in analytical solutions
  - 1x1 ODE systems use `cmath.exp()` for complex eigenvalues
  - 2x2 systems fall back to matrix exponential for complex eigenvalues
- **Performance improvements**
  - Matrix exponentiation now uses binary exponentiation (O(log n) instead of O(n))
  - Extracted helper methods to eliminate code duplication
  - `__truediv__` optimized using `_scalar_multiply(1.0 / scalar)`
- **Code quality**
  - Added `DEFAULT_TOLERANCE = 1e-10` constant across modules
  - Replaced all hardcoded tolerance values with named constant
  - Removed redundant `__rmatmul__` method
  - Extracted `_scalar_multiply()` and `_matrix_multiply()` helpers
  - Reduced code duplication by ~35 lines

### Fixed
- Eigenvector normalization error when handling complex eigenvectors
- ODE solver compatibility with complex eigenvalues
- Matrix power operation efficiency

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

[0.3.0]: https://github.com/njryan-boou/matpy/releases/tag/v0.3.0
[0.2.0]: https://github.com/njryan-boou/matpy/releases/tag/v0.2.0
[0.1.0]: https://github.com/njryan-boou/matpy/releases/tag/v0.1.0
