# MatPy

A comprehensive Python library for vector and matrix operations with a clean, Pythonic API. MatPy provides an intuitive interface for linear algebra operations, making it perfect for educational purposes, scientific computing, and mathematical applications.

## Features

### 🎯 Core Components

- **Vector Class**: N-dimensional vectors with full operator support (2D, 3D, and beyond)
- **Matrix Class**: Dynamic-size matrices with comprehensive operations
- **Linear System Solvers**: Gaussian elimination, LU decomposition, Cramer's rule, least squares
- **ODE Solvers**: Systems of differential equations (homogeneous and non-homogeneous)
- **Coordinate Systems**: Convert between Cartesian, Polar, Spherical, and Cylindrical coordinates
- **Visualization**: Rich plotting capabilities with matplotlib (optional)
- **Custom Error Handling**: Descriptive exceptions for better debugging
- **Pure Python Core**: No required dependencies for core functionality

### ⚡ Vector Operations

- **Basic Operations**: `+`, `-`, `*`, `/` with full operator support
- **Vector Products**: Dot product, cross product (3D)
- **Magnitude & Normalization**: Length calculation and unit vectors
- **Advanced Operations**: Projection, rejection, reflection
- **Geometric Functions**: Angle calculation, distance, component-wise min/max
- **Interpolation**: Linear interpolation (lerp), clamping
- **Parallel & Perpendicular Tests**: Check vector relationships
- **N-Dimensional Support**: Works with 2D, 3D, 4D, and higher dimensions
- **Full Python Protocols**: Iteration, indexing, equality, hashing, etc.

### 🔢 Matrix Operations

- **Arithmetic**: Element-wise operations, matrix multiplication (`@`), power (`**`)
- **Basic Operations**: Transpose, trace, determinant, rank
- **Advanced Operations**: Inverse, adjugate, cofactor, matrix exponential
- **Matrix Creation**: Zeros, ones, identity, diagonal, from rows/columns
- **Matrix Properties**: Symmetric, orthogonal, singular, triangular, diagonal
- **Hadamard Product**: Element-wise multiplication
- **Kronecker Product**: Tensor product of matrices
- **Row Operations**: Row echelon form, reduced row echelon form
- **Concatenation**: Horizontal and vertical matrix joining

### 🧮 Linear Algebra Solvers

- **Linear Systems**: Solve Ax = b using multiple methods
### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/matpy.git
cd matpy

# Install core library
pip install .

# Or install in development mode
pip install -e .

# Install with visualization support
pip install .[viz]

# Install with all development tools
pip install .[dev]
```

### From PyPI (Coming Soon)

```bash
# Core installation
pip install matpy-linalg

# With visualization support
pip install matpy-linalg[viz]
### 🌐 Coordinate Systems

- **2D Conversions**: Cartesian ↔ Polar ↔ Complex
- **3D Conversions**: Cartesian ↔ Spherical ↔ Cylindrical
- **Easy API**: `VectorCoordinates` class with intuitive methods

### 📊 Visualization (Optional)

- **Vector Plotting**: 2D and 3D vector arrows with labels
- **Vector Fields**: Visualize 2D vector fields
- **Matrix Heatmaps**: Color-coded matrix visualization
- **Transformations**: Before/after visualization of linear transformations
- **Coordinate Systems**: Side-by-side comparison of coordinate representations
- **Customizable**: Colors, labels, titles, and full matplotlib control

## Installation
.vector.core import Vector
from matpy.vector import ops

# Create vectors (n-dimensional)
v1 = Vector(3, 4, 5)      # 3D vector
v2 = Vector(1, 2, 3)      # 3D vector
v_2d = Vector(3, 4)       # 2D vector
v_4d = Vector(1, 2, 3, 4) # 4D vector

# Arithmetic operations
v3 = v1 + v2          # Vector addition
v4 = v1 - v2          # Vector subtraction
v5 = v1 * 2           # Scalar multiplication
v6 = v1 / 2           # Scalar division

# Vector operations
dot_product = v1.dot(v2)           # Dot product
cross_product = v1.cross(v2)       # Cross product (3D only)
magnitude = v1.magnitude()         # Magnitude/length
normalized = v1.normalize()        # Unit vector

# Advanced operations
angle = ops.angle_between(v1, v2)  # Angle in radians
projection = ops.projection(v1, v2)
rejection = ops.rejection(v1, v2)
reflected = ops.reflect(v1, Vector(0, 1, 0))
interpolated = ops.lerp(v1, v2, 0.5)  # 50% between v1 and v2
distance = ops.distance(v1, v2)

# Component operations
min_vec = ops.component_min(v1, v2)  # Element-wise minimum
max_vec = ops.component_max(v1, v2)  # Element-wise maximum
clamped = ops.clamp(v1, -1, 1)       # Clamp all components

# Vector tests
parallel = ops.is_parallel(v1, v2)
perpendicular = ops.is_perpendicular(v1, v2)

# Properties for 2D/3D vectors
print(v1.x, v1.y, v1.z)  # Access x, y, z components

# Python protocols
print(v1)              # (3.0, 4.0, 5.0)
print(repr(v1))        # <3.0, 4.0, 5.0>
print(len(v1))         # 3
print(v1[0])           # 3.0alar multiplication
v6 = v1 / 2           # Scalar division

# Vector operations
dot_product = v1.dot(v2)           # Dot product
cross_product = v1.cross(v2)       # Cross product
magnitude = v1.magnitude()         # Magnitude/length
normalized = v1.normalize()        # Unit vector

# Advanced operations
from matpy.matrix import ops

# Create matrices
m1 = Matrix(2, 2, [[1, 2], [3, 4]])
m2 = Matrix(2, 2, [[5, 6], [7, 8]])

# Matrix creation utilities
zeros = ops.zeros(3, 3)              # 3x3 zero matrix
ones = ops.ones(2, 4)                # 2x4 matrix of ones
identity = ops.identity(3)            # 3x3 identity matrix
diagonal = ops.diagonal([1, 2, 3])    # 3x3 diagonal matrix
from_rows = ops.from_rows([[1, 2], [3, 4]])
from_cols = ops.from_columns([[1, 3], [2, 4]])

# Arithmetic operations
m3 = m1 + m2          # Matrix addition
m4 = m1 - m2          # Matrix subtraction
m5 = m1 * 3           # Scalar multiplication
m6 = m1 @ m2          # Matrix multiplication (@ operator)
m7 = m1 ** 3          # Matrix power

# Matrix operations
transposed = m1.transpose()
determinant = m1.determinant()
inverted = m1.inverse()
trace = m1.trace()
rank = m1.rank()

# Advanced operations
adjugate = m1.adjugate()
cofactor = m1.cofactor(0, 1)
hadamard = ops.hadamard_product(m1, m2)  # Element-wise multiplication
kronecker = ops.kronecker_product(m1, m2)
rref = ops.reduced_row_echelon_form(m1)

# Matrix properties
is_square = m1.is_square()
is_symmetric = m1.is_symmetric()
is_singular = m1.is_singular()
is_invertible = m1.is_invertible()
is_orthogonal = m1.is_orthogonal()
is_diagonal = ops.is_diagonal(m1)
is_identity = ops.is_identity(m1)

# Concatenation
h_concat = ops.concatenate_horizontal(m1, m2)
v_concat = ops.concatenate_vertical(m1, m2)

# Python protocols
print(m1)              # Formatted matrix output
print(f"{m1:.2f}")     # Formatted to 2 decimal places
for value in m1:       # Iterate over all elements
    print(value)
```

### Linear Systemscomprehensive demonstrations:

### Vector Examples
- `vector_arithmatic.py` - Vector arithmetic operations
- `python_methods.py` - Python dunder methods and protocols

### Matrix Examples
- `matrix_examples.py` - Complete matrix operations showcase
  - Matrix creation and basic operations
  - Linear algebra operations (determinant, inverse, etc.)
  - Linear systems solving
  - Least squares fitting
  - Advanced operations (Hadamard, Kronecker products)
  - Differential equations

### Visualization Examples
- `visualization_examples.py` - Complete visualization demonstrations
  - 2D and 3D vector plotting
  - Vector fields
  - Matrix heatmaps and grids
  - Linear transformations (rotations, scaling, shearing)
  - Coordinate system comparisons
  - Vector operations visualization

# Solve Ax = b
A = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = [8, -11, -3]

# Multiple solution methods
x = solve.solve_linear_system(A, b)      # Gaussian elimination
x_lu = solve.solve_lu(A, b)               # LU decomposition
x_cramer = solve.solve_cramer(        # Custom exceptions
│       ├── core/                     # Core utilities
│       │   ├── __init__.py
│       │   ├── utils.py              # Utility functions (formatting, math, etc.)
│       │   └── validate.py           # Validation functions
│       ├── vector/                   # Vector implementation
│       │   ├── __init__.py
│       │   ├── core.py               # N-dimensional Vector class
│       │   ├── ops.py                # Vector operations and functions
│       │   └── coordinates.py        # Coordinate system conversions
│       ├── matrix/                   # Matrix implementation
│       │   ├── __init__.py
│       │   ├── core.py               # Matrix class
│       │   ├── ops.py                # Matrix operations and utilities
│       │   └── solve.py              # Linear systems and ODE solvers
│       └── visualization/            # Visualization tools (optional)
│           ├── __init__.py
│           ├── vector_plot.py        # Vector plotting functions
│           ├── matrix_plot.py        # Matrix visualization
│           └── coordinate_plot.py    # Coordinate system plots
├── tests/                            # Comprehensive test suite
│   ├── test_vector_core.py           # Vector class tests
│   ├── test_vector_ops.py            # Vector operations tests
│   ├── test_matrix_core.py           # Matrix class tests
│   ├── test_matrix_ops.py            # Matrix operations tests
│   └── test_matrix_solve.py          # Solver tests
├── examples/                         # Example scripts
│   ├── vector_arithmatic.py
│   ├── python_methods.py
│   ├── matrix_examples.py
│   └── visualization_examples.py
├── pyproject.toml                    # Project configuration
└── README.md         
from matpy.matrix.core import Matrix
from matpy.matrix import solve

# Solve dx/dt = Ax
A = Matrix(2, 2, [[-2, 1], [1, -2]])
x0 = [1, 0]  # Initial conditions
t = 1.0      # Time

# Homogeneous system
x_t = solve.solve_linear_ode_system_homogeneous(A, x0, t)

# Non-homogeneous system: dx/dt = Ax + b(t)
def b_func(t):
    return [t, 0]

x_t_nh = solve.solve_linear_ode_system_nonhomogeneous(A, x0, b_func, t)
*args)` - Create an n-dimensional vector
  - `Vector()` - Creates 3D zero vector (0, 0, 0)
  - `Vector(x, y)` - 2D vector
  - `Vector(x, y, z)` - 3D vector
  - `Vector(x, y, z, w, ...)` - N-dimensional vector

**Properties (2D/3D):**
- `x`, `y`, `z` - Access first three components
- `components` - Tuple of all components

**Methods:**
- `dot(other)` - Dot product
- `cross(other)` - Cross product (3D only)
- `magnitude()` - Calculate magnitude
- `normalize()` - Return unit vector

**Operators:**
- `+`, `-`, `*`, `/` - Arithmetic operations
- `==`, `!=` - Equality comparison
- `abs()` - Magnitude
- `len()` - Number of dimensions
- `[]` - Index access
- `in` - Membership test
- `iter()` - Iteration support
- `bool()` - True if non-zero
- `round(n)` - Round componentsv_2d)

# Convert to polar
r, theta = coords_2d.to_polar()
print(f"Polar: r={r}, θ={theta}")

# Create from polar
v_from_polar = VectorCoordinates.from_polar(5, math.pi/4)

# Complex representation
z = coords_2d.to_complex()  # 3 + 4j
v_from_complex = VectorCoordinates.from_complex(3 + 4j)

# 3D Coordinate conversions
v_3d = Vector(1, 1, math.sqrt(2))
coords_3d = VectorCoordinates(v_3d)

# Spherical coordinates
r, theta, phi = coords_3d.to_spherical()
v_from_spherical = VectorCoordinates.from_spherical(2, math.pi/4, math.pi/4)

# Cylindrical coordinates
rho, phi, z = coords_3d.to_cylindrical()
v_from_cylindrical = VectorCoordinates.from_cylindrical(math.sqrt(2), math.pi/4, math.sqrt(2))

# Generic conversion
polar_coords = coords_2d.convert('polar')
spherical_coords = coords_3d.convert('spherical')
```

### Visualization

```python
from matpy.vector.core import Vector
from matpy.matrix.core import Matrix
from matpy.visualization import (
    plot_vectors_2d, plot_vectors_3d,
    plot_transformation_2d, plot_matrix_heatmap
)
import math

# Plot 2D vectors
v1 = Vector(2, 3)
v2 = Vector(-1, 2)
plot_vectors_2d([v1, v2], labels=['v1', 'v2'], title="My Vectors")

# Plot 3D vectors
i = Vector(1, 0, 0)
j = Vector(0, 1, 0)
k = Vector(0, 0, 1)
plot_vectors_3d([i, j, k], labels=['i', 'j', 'k'], colors=['red', 'green', 'blue'])

# Visualize transformation
angle = math.pi / 4  # 45 degrees
rotation = Matrix(2, 2, [
    [math.cos(angle), -math.sin(angle)],
    [math.sin(angle), math.cos(angle)]
])
plot_transformation_2d(rotation, title="45° Rotation")

# Matrix heatmap
m = Matrix(4, 4, [[i+j for j in range(4)] for i in range(4)])
plot_matrix_heatmap(m, title="Sample Matrix"e()
rank = m1.rank()

# Matrix properties
is_square = m1.is_square()
is_symmetric = m1.is_symmetric()
is_singular = m1.is_singular()
is_invertible = m1.is_invertible()

# Advanced operations
adjugate = m1.adjugate()
cofactor = m1.cofactor(0, 1)

# Python protocols
print(m1)              # Formatted matrix output
print(f"{m1:.2f}")     # Formatted to 2 decimal places
for value in m1:       # Iterate over all elements
    print(value)
```

## Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
python tests/test_vector_core.py
python tests/test_vector_ops.py

# Run with unittest
python -m unittest discover tests

# Run with pytest (if installed)
pytest tests/
```

## Examples

Check out the `examples/` directory for more detailed examples:

- `Vector_example.py` - Comprehensive vector operations showcase
- `vector_arithmatic.py` - Vector arithmetic demonstrations
- `python_methods.py` - Python dunder methods examples

## Project Structure

```
matpy/
├── src/
│   └── matpy/
│       ├── __init__.py
│       ├── version.py
│       ├── error.py          # Custom exceptions
│       ├── core/             # Core utilities
│       │   ├── utils.py
│       │   └── validate.py
│       ├── vector/           # Vector implementation
│       │   ├── core.py       # Vector class
│       │   └── ops.py        # Vector operations
│       ├── matrix/           # Matrix implementation
│       │   └── core.py       # Matrix class
│       └── visualization/    # Visualization tools
├── tests/                    # Test suite
│   ├── test_vector_core.py
│   ├── test_vector_ops.py
│   └── run_tests.py
├── examples/                 # Example scripts
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Custom Error Handling

MatPy provides descriptive custom exceptions for better error handling:

```python
from matpy.error import (
    ValidationError,
  - If `data=None`, creates zero matrix
  - `data` should be a 2D list: `[[row1], [row2], ...]`

**Methods:**
- `transpose()` - Matrix transpose
- `determinant()` - Calculate determinant (square matrices)
- `inverse()` - Matrix inverse (non-singular matrices)
- `adjugate()` - Adjugate (adjoint) matrix
- `trace()` - Sum of diagonal elements
- `rank()` - Matrix rank
- `cofactor(row, col)` - Cofactor at position
- `is_square()` - Check if square
- `is_symmetric()` - Check if M = M^T
- `is_singular()` - Check if determinant ≈ 0
- `is_invertible()` - Check if non-singular
- `is_orthogonal()` - Check if M^T M = I

**Operators:**
- `+`, `-` - Matrix addition/subtraction
- `*` - Scalar or matrix multiplication
- `/` - Scalar division
- `@` - Matrix multiplication (preferred)
- `**` - Matrix power
- `==`, `!=`, `<`, `>`, `<=`, `>=` - Comparisons
- `-M` - Negation
- `abs()` - Frobenius norm
- `len()` - Total number of elements
- `[i]` - Row access
- `[i] = row` - Row assignment
- `iter()` - Iteration over elements
- `bool()` - True if has non-zero elements
- `round(n)` - Round all elements

### Matrix Operations Module

**Creation Functions:**
- `zeros(rows, cols)` - Zero matrix
- `ones(rows, cols)` - Matrix of ones
- `identity(size)` - Identity matrix
- `diagonal(values)` - Diagonal matrix
- Install with visualization support
pip install -e ".[viz]"

# Run tests
python -m unittest discover -s tests -p "test_*.py"

# Run tests with pytest (if installed)
pytest tests/

# Format code
blaRecent Updates

✅ **v0.1.0 (Current)**
- N-dimensional vector support (2D, 3D, 4D+)
- Complete linear systems solver suite
- ODE solver for systems of differential equations
- Coordinate system conversions (Polar, Spherical, Cylindrical, Complex)
- Comprehensive visualization module with matplotlib
- Advanced matrix operations (Hadamard, Kronecker products, RREF)
- Centralized validation and utility modules
- 300+ unit tests with comprehensive coverage

## Future Enhancements

- [ ] Additional matrix decompositions (QR, SVD, Cholesky)
- [ ] Sparse matrix support
- [ ] Higher-dimensional tensors
- [ ] Complex number support for matrices
- [ ] Performance optimizations with optional NumPy backend
- [ ] Additional numerical ODE solvers
- [ ] Eigenvalue/eigenvector computation for larger matrices
- [ ] Interactive visualization widgetsm2)` - Join top-to-bottom

**Property Tests:**
- `is_diagonal(m)`, `is_identity(m)`, `is_upper_triangular(m)`, `is_lower_triangular(m)`

### Linear Algebra Solvers

**Linear Systems:**
- `solve_linear_system(A, b)` - Gaussian elimination
- `solve_lu(A, b)` - Using LU decomposition
- `solve_cramer(A, b)` - Cramer's rule
- `solve_least_squares(A, b)` - Least squares solution
- `lu_decomposition(A)` - Returns (L, U) matrices

**Differential Equations:**
- `solve_linear_ode_system_homogeneous(A, x0, t)` - Solve dx/dt = Ax
- `solve_linear_ode_system_nonhomogeneous(A, x0, b_func, t)` - Solve dx/dt = Ax + b(t)
- `matrix_exponential(A, t)` - Compute e^(At)
- `euler_method(A, x0, t_final, steps)` - Numerical ODE solver
- `runge_kutta_4(A, x0, t_final, steps)` - RK4 numerical solver

### Coordinate Systems

**VectorCoordinates Class:**
- 2D: `to_polar()`, `from_polar()`, `to_complex()`, `from_complex()`
- 3D: `to_spherical()`, `from_spherical()`, `to_cylindrical()`, `from_cylindrical()`
- Generic: `convert(system)`, `convert_from(system, *coords)`

### Visualization Functions

**Vector Plotting:**
- `plot_vector_2d(v, ...)` - Plot single 2D vector
- `plot_vectors_2d(vectors, ...)` - Plot multiple 2D vectors
- `plot_vector_3d(v, ...)` - Plot single 3D vector
- `plot_vectors_3d(vectors, ...)` - Plot multiple 3D vectors
- `plot_vector_field_2d(func, ...)` - Plot 2D vector field

**Matrix Visualization:**
- `plot_matrix_heatmap(m, ...)` - Color-coded matrix
- `plot_matrix_grid(m, ...)` - Bar chart representation
- `plot_transformation_2d(m, ...)` - Visualize 2D transformation
- `plot_transformation_3d(m, ...)` - Visualize 3D transformation

**Coordinate Plotting:**
- `plot_coordinate_systems_2d(v)` - Cartesian vs Polar
- `plot_coordinate_systems_3d(v)` - Cartesian vs Spherical vs Cylindrical
- `Vector(x=0, y=0, z=0)` - Create a 3D vector

**Methods:**
- `dot(other)` - Dot product
- `cross(other)` - Cross product
- `magnitude()` - Calculate magnitude
- `normalize()` - Return unit vector

**Operators:**
- `+`, `-`, `*`, `/` - Arithmetic operations
- `==`, `!=` - Equality comparison
- `abs()` - Magnitude
- `len()` - Always returns 3
- `[]` - Index access (0, 1, 2)
- `in` - Membership test
- `iter()` - Iteration support

### Matrix Class

**Constructor:**
- `Matrix(rows, cols, data=None)` - Create a matrix

**Methods:**
- `transpose()` - Matrix transpose
- `determinant()` - Calculate determinant
- `inverse()` - Matrix inverse
- `adjugate()` - Adjugate matrix
- `trace()` - Sum of diagonal elements
- `rank()` - Matrix rank
- `cofactor(row, col)` - Cofactor calculation
- `is_square()` - Check if square
- `is_symmetric()` - Check if symmetric
- `is_singular()` - Check if singular
- `is_invertible()` - Check if invertible

**Operators:**
- `+`, `-`, `*`, `/` - Arithmetic operations
- `@` - Matrix multiplication
- `**` - Matrix power
- `==`, `!=`, `<`, `>`, `<=`, `>=` - Comparisons
- `abs()` - Frobenius norm
- `len()` - Total number of elements
- `[]` - Row access
- `iter()` - Iteration support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=matpy

# Format code
black src/ tests/

# Type checking
mypy src/
```

## Future Enhancements

- [ ] Matrix decomposition (LU, QR, SVD)
- [ ] Sparse matrix support
- [ ] Higher-dimensional tensors
- [ ] Visualization module with matplotlib
- [ ] Performance optimizations with NumPy backend (optional)
- [ ] Additional matrix operations (Cholesky, Schur, etc.)
- [ ] Complex number support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by NumPy and other linear algebra libraries
- Built with pure Python for educational purposes
- Comprehensive test coverage for reliability

## Contact

Noah Ryan - njryan2005@gmail.com

Project Link: [https://github.com/yourusername/matpy](https://github.com/yourusername/matpy)
