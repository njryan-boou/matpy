---
layout: default
title: API Reference
---

# API Reference

[← Back to Home](index.md)

Complete API reference for MatPy.

## Table of Contents

- [Vector Class](#vector-class)
- [Vector Operations](#vector-operations)
- [Vector Coordinates](#vector-coordinates)
- [Matrix Class](#matrix-class)
- [Matrix Operations](#matrix-operations)
- [Matrix Solvers](#matrix-solvers)
- [Visualization](#visualization)
- [Error Classes](#error-classes)

---

## Vector Class

### `Vector(*args)`

Create an n-dimensional vector.

**Parameters:**
- `*args` (float): Components of the vector

**Returns:**
- `Vector`: A new vector instance

**Examples:**
```python
v_2d = Vector(3, 4)           # 2D vector
v_3d = Vector(1, 2, 3)        # 3D vector
v_4d = Vector(1, 2, 3, 4)     # 4D vector
v_zero = Vector()             # Zero vector (0, 0, 0)
```

### Properties

- `x`, `y`, `z`: First three components (for 2D/3D vectors)
- `components`: Tuple of all components

### Methods

#### `dot(other: Vector) -> float`
Calculate dot product.

#### `cross(other: Vector) -> Vector`
Calculate cross product (3D only).

#### `magnitude() -> float`
Calculate vector magnitude/length.

#### `normalize() -> Vector`
Return unit vector in same direction.

### Operators

- `+`, `-`: Vector addition/subtraction
- `*`, `/`: Scalar multiplication/division
- `==`, `!=`: Equality comparison
- `abs()`: Magnitude
- `len()`: Number of dimensions
- `[]`: Index access
- `iter()`: Iteration support

---

## Vector Operations

Module: `matpy.vector.ops`

### Geometric Operations

#### `angle_between(v1: Vector, v2: Vector) -> float`
Calculate angle between vectors in radians.

#### `distance(v1: Vector, v2: Vector) -> float`
Calculate Euclidean distance.

#### `projection(v1: Vector, v2: Vector) -> Vector`
Project v1 onto v2.

#### `rejection(v1: Vector, v2: Vector) -> Vector`
Calculate rejection of v1 from v2.

#### `reflect(v: Vector, normal: Vector) -> Vector`
Reflect vector across surface defined by normal.

### Interpolation

#### `lerp(v1: Vector, v2: Vector, t: float) -> Vector`
Linear interpolation between vectors.

#### `clamp(v: Vector, min_val: float, max_val: float) -> Vector`
Clamp all components to range.

### Component Operations

#### `component_min(v1: Vector, v2: Vector) -> Vector`
Element-wise minimum.

#### `component_max(v1: Vector, v2: Vector) -> Vector`
Element-wise maximum.

### Tests

#### `is_parallel(v1: Vector, v2: Vector, tolerance: float = 1e-10) -> bool`
Check if vectors are parallel.

#### `is_perpendicular(v1: Vector, v2: Vector, tolerance: float = 1e-10) -> bool`
Check if vectors are perpendicular.

---

## Vector Coordinates

Module: `matpy.vector.coordinates`

### `VectorCoordinates(vector: Vector)`

### 2D Methods

#### `to_polar() -> Tuple[float, float]`
Convert to polar coordinates (r, θ).

#### `from_polar(r: float, theta: float) -> Vector`
Create vector from polar coordinates.

#### `to_complex() -> complex`
Convert to complex number.

#### `from_complex(z: complex) -> Vector`
Create vector from complex number.

### 3D Methods

#### `to_spherical() -> Tuple[float, float, float]`
Convert to spherical coordinates (r, θ, φ).

#### `from_spherical(r: float, theta: float, phi: float) -> Vector`
Create vector from spherical coordinates.

#### `to_cylindrical() -> Tuple[float, float, float]`
Convert to cylindrical coordinates (ρ, φ, z).

#### `from_cylindrical(rho: float, phi: float, z: float) -> Vector`
Create vector from cylindrical coordinates.

### Generic

#### `convert(system: str) -> Union[Tuple, complex]`
Convert to specified coordinate system.

---

## Matrix Class

### `Matrix(rows: int, cols: int, data: Optional[List[List[float]]] = None)`

Create a matrix.

**Parameters:**
- `rows` (int): Number of rows
- `cols` (int): Number of columns
- `data` (List[List[float]], optional): Initial data

**Examples:**
```python
m = Matrix(2, 2, [[1, 2], [3, 4]])
zeros = Matrix(3, 3)  # Zero matrix
```

### Methods

#### `transpose() -> Matrix`
Return transpose.

#### `determinant() -> float`
Calculate determinant (square matrices only).

#### `inverse() -> Matrix`
Calculate matrix inverse.

#### `adjugate() -> Matrix`
Calculate adjugate matrix.

#### `trace() -> float`
Sum of diagonal elements.

#### `rank() -> int`
Calculate matrix rank.

#### `cofactor(row: int, col: int) -> float`
Calculate cofactor at position.

### Property Methods

#### `is_square() -> bool`
Check if matrix is square.

#### `is_symmetric() -> bool`
Check if matrix equals its transpose.

#### `is_singular() -> bool`
Check if determinant ≈ 0.

#### `is_invertible() -> bool`
Check if matrix has an inverse.

#### `is_orthogonal() -> bool`
Check if M^T M = I.

### Operators

- `+`, `-`: Matrix addition/subtraction
- `*`: Scalar or matrix multiplication
- `/`: Scalar division
- `@`: Matrix multiplication (preferred)
- `**`: Matrix power
- `abs()`: Frobenius norm

---

## Matrix Operations

Module: `matpy.matrix.ops`

### Creation Functions

#### `zeros(rows: int, cols: int) -> Matrix`
Create zero matrix.

#### `ones(rows: int, cols: int) -> Matrix`
Create matrix of ones.

#### `identity(size: int) -> Matrix`
Create identity matrix.

#### `diagonal(values: List[float]) -> Matrix`
Create diagonal matrix.

#### `from_rows(rows: List[List[float]]) -> Matrix`
Create from row lists.

#### `from_columns(cols: List[List[float]]) -> Matrix`
Create from column lists.

### Advanced Operations

#### `hadamard_product(m1: Matrix, m2: Matrix) -> Matrix`
Element-wise multiplication.

#### `kronecker_product(m1: Matrix, m2: Matrix) -> Matrix`
Tensor product.

#### `row_echelon_form(m: Matrix) -> Matrix`
Convert to REF.

#### `reduced_row_echelon_form(m: Matrix) -> Matrix`
Convert to RREF.

#### `concatenate_horizontal(m1: Matrix, m2: Matrix) -> Matrix`
Join matrices side-by-side.

#### `concatenate_vertical(m1: Matrix, m2: Matrix) -> Matrix`
Join matrices top-to-bottom.

### Property Tests

#### `is_diagonal(m: Matrix) -> bool`
#### `is_identity(m: Matrix) -> bool`
#### `is_upper_triangular(m: Matrix) -> bool`
#### `is_lower_triangular(m: Matrix) -> bool`

---

## Matrix Solvers

Module: `matpy.matrix.solve`

### Linear Systems

#### `solve_linear_system(A: Matrix, b: List[float]) -> List[float]`
Solve Ax = b using Gaussian elimination.

#### `solve_lu(A: Matrix, b: List[float]) -> List[float]`
Solve using LU decomposition.

#### `solve_cramer(A: Matrix, b: List[float]) -> List[float]`
Solve using Cramer's rule.

#### `solve_least_squares(A: Matrix, b: List[float]) -> List[float]`
Least squares solution for overdetermined systems.

#### `lu_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]`
LU decomposition, returns (L, U).

### Differential Equations

#### `solve_linear_ode_system_homogeneous(A: Matrix, x0: List[float], t: float) -> List[float]`
Solve dx/dt = Ax.

#### `solve_linear_ode_system_nonhomogeneous(A: Matrix, x0: List[float], b_func: Callable, t: float) -> List[float]`
Solve dx/dt = Ax + b(t).

#### `matrix_exponential(A: Matrix, t: float = 1.0) -> Matrix`
Compute e^(At).

#### `euler_method(A: Matrix, x0: List[float], t_final: float, steps: int) -> List[float]`
Euler numerical solver.

#### `runge_kutta_4(A: Matrix, x0: List[float], t_final: float, steps: int) -> List[float]`
RK4 numerical solver.

---

## Visualization

Module: `matpy.visualization`

### Vector Plotting

#### `plot_vector_2d(v: Vector, **kwargs)`
Plot single 2D vector.

#### `plot_vectors_2d(vectors: List[Vector], labels: List[str] = None, **kwargs)`
Plot multiple 2D vectors.

#### `plot_vector_3d(v: Vector, **kwargs)`
Plot single 3D vector.

#### `plot_vectors_3d(vectors: List[Vector], labels: List[str] = None, **kwargs)`
Plot multiple 3D vectors.

#### `plot_vector_field_2d(func: Callable, x_range: Tuple, y_range: Tuple, **kwargs)`
Plot 2D vector field.

### Matrix Plotting

#### `plot_matrix_heatmap(m: Matrix, **kwargs)`
Color-coded matrix visualization.

#### `plot_matrix_grid(m: Matrix, **kwargs)`
Bar chart representation.

#### `plot_transformation_2d(m: Matrix, **kwargs)`
Visualize 2D transformation.

#### `plot_transformation_3d(m: Matrix, **kwargs)`
Visualize 3D transformation.

### Coordinate Plotting

#### `plot_coordinate_systems_2d(v: Vector, **kwargs)`
Compare Cartesian and Polar.

#### `plot_coordinate_systems_3d(v: Vector, **kwargs)`
Compare Cartesian, Spherical, and Cylindrical.

---

## Error Classes

Module: `matpy.error`

### `MatPyError`
Base exception class.

### `VectorDimensionError`
Raised when vector dimensions don't match.

### `MatrixDimensionError`
Raised when matrix dimensions are invalid.

### `InvalidOperationError`
Raised for invalid operations.

### `SingularMatrixError`
Raised when operating on singular matrices.

---

[← Back to Home](index.md)
