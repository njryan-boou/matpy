---
layout: default
title: Tutorials
---

# Tutorials

[← Back to Home](index.md)

Learn how to use MatPy with these step-by-step tutorials.

## Table of Contents

1. [Getting Started with Vectors](#getting-started-with-vectors)
2. [Working with Matrices](#working-with-matrices)
3. [Solving Linear Systems](#solving-linear-systems)
4. [Coordinate System Conversions](#coordinate-system-conversions)
5. [Visualization](#visualization)
6. [Advanced Topics](#advanced-topics)

---

## Getting Started with Vectors

### Creating Vectors

MatPy supports n-dimensional vectors:

```python
from matpy.vector.core import Vector

# Create vectors of any dimension
v_2d = Vector(3, 4)           # 2D vector
v_3d = Vector(1, 2, 3)        # 3D vector
v_4d = Vector(1, 2, 3, 4)     # 4D vector
v_zero = Vector()             # Zero vector (0, 0, 0)
```

### Basic Vector Operations

```python
v1 = Vector(3, 4, 5)
v2 = Vector(1, 2, 3)

# Arithmetic
v_sum = v1 + v2              # (4, 6, 8)
v_diff = v1 - v2             # (2, 2, 2)
v_scaled = v1 * 2            # (6, 8, 10)
v_divided = v1 / 2           # (1.5, 2, 2.5)

# Vector products
dot = v1.dot(v2)             # 26
cross = v1.cross(v2)         # (-2, -4, 2) - 3D only

# Properties
magnitude = v1.magnitude()    # 7.071...
unit = v1.normalize()        # Unit vector
```

### Advanced Vector Operations

```python
from matpy.vector import ops

# Angle between vectors
angle = ops.angle_between(v1, v2)  # Radians

# Projection and rejection
proj = ops.projection(v1, v2)      # Project v1 onto v2
rej = ops.rejection(v1, v2)        # Rejection of v1 from v2

# Interpolation
midpoint = ops.lerp(v1, v2, 0.5)   # Linear interpolation

# Distance
dist = ops.distance(v1, v2)        # Euclidean distance

# Component operations
min_vec = ops.component_min(v1, v2)  # Element-wise minimum
max_vec = ops.component_max(v1, v2)  # Element-wise maximum
clamped = ops.clamp(v1, -1, 1)       # Clamp all components

# Tests
parallel = ops.is_parallel(v1, v2)
perpendicular = ops.is_perpendicular(v1, v2)
```

---

## Working with Matrices

### Creating Matrices

```python
from matpy.matrix.core import Matrix
from matpy.matrix import ops

# Direct creation
m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])

# Special matrices
zeros = ops.zeros(3, 3)              # Zero matrix
ones = ops.ones(2, 4)                # Matrix of ones
identity = ops.identity(3)            # Identity matrix
diagonal = ops.diagonal([1, 2, 3])    # Diagonal matrix

# From rows or columns
m_rows = ops.from_rows([[1, 2], [3, 4]])
m_cols = ops.from_columns([[1, 3], [2, 4]])
```

### Matrix Operations

```python
m1 = Matrix(2, 2, [[1, 2], [3, 4]])
m2 = Matrix(2, 2, [[5, 6], [7, 8]])

# Basic operations
m_sum = m1 + m2              # Matrix addition
m_diff = m1 - m2             # Matrix subtraction
m_scaled = m1 * 3            # Scalar multiplication
m_product = m1 @ m2          # Matrix multiplication
m_power = m1 ** 3            # Matrix power

# Matrix properties
transposed = m1.transpose()
determinant = m1.determinant()
inverse = m1.inverse()
trace = m1.trace()
rank = m1.rank()

# Advanced operations
adjugate = m1.adjugate()
hadamard = ops.hadamard_product(m1, m2)  # Element-wise
kronecker = ops.kronecker_product(m1, m2)  # Tensor product
rref = ops.reduced_row_echelon_form(m1)
```

---

## Solving Linear Systems

### Gaussian Elimination

```python
from matpy.matrix.core import Matrix
from matpy.matrix import solve

# Solve Ax = b
A = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = [8, -11, -3]

# Gaussian elimination (default method)
x = solve.solve_linear_system(A, b)
print(f"Solution: {x}")  # [2, 3, -1]
```

### LU Decomposition

```python
# Using LU decomposition
x_lu = solve.solve_lu(A, b)

# Get L and U matrices
L, U = solve.lu_decomposition(A)
print(f"L = {L}")
print(f"U = {U}")
```

### Other Methods

```python
# Cramer's rule (for small systems)
x_cramer = solve.solve_cramer(A, b)

# Least squares (for overdetermined systems)
A_over = Matrix(4, 2, [[1, 1], [2, 1], [3, 1], [4, 1]])
b_over = [2, 3, 5, 4]
x_ls = solve.solve_least_squares(A_over, b_over)
```

### Differential Equations

```python
# Solve dx/dt = Ax
A = Matrix(2, 2, [[-2, 1], [1, -2]])
x0 = [1, 0]  # Initial conditions
t = 1.0

# Exact solution
x_t = solve.solve_linear_ode_system_homogeneous(A, x0, t)

# With forcing function: dx/dt = Ax + b(t)
def b_func(t):
    return [t, 0]

x_t_nh = solve.solve_linear_ode_system_nonhomogeneous(A, x0, b_func, t)

# Numerical methods
euler = solve.euler_method(A, x0, t_final=2.0, steps=100)
rk4 = solve.runge_kutta_4(A, x0, t_final=2.0, steps=100)
```

---

## Coordinate System Conversions

### 2D Conversions

```python
from matpy.vector.core import Vector
from matpy.vector.coordinates import VectorCoordinates
import math

v = Vector(3, 4)
coords = VectorCoordinates(v)

# Cartesian to Polar
r, theta = coords.to_polar()
print(f"Polar: r={r}, θ={theta}")

# Polar to Cartesian
v_from_polar = VectorCoordinates.from_polar(5, math.pi/4)

# Complex number representation
z = coords.to_complex()  # 3 + 4j
v_from_complex = VectorCoordinates.from_complex(3 + 4j)
```

### 3D Conversions

```python
v = Vector(1, 1, math.sqrt(2))
coords = VectorCoordinates(v)

# Spherical coordinates
r, theta, phi = coords.to_spherical()
v_sph = VectorCoordinates.from_spherical(2, math.pi/4, math.pi/4)

# Cylindrical coordinates
rho, phi, z = coords.to_cylindrical()
v_cyl = VectorCoordinates.from_cylindrical(math.sqrt(2), math.pi/4, 1.414)

# Generic conversion
polar = coords.convert('polar')
spherical = coords.convert('spherical')
```

---

## Visualization

### Vector Plotting

```python
from matpy.vector.core import Vector
from matpy.visualization import plot_vectors_2d, plot_vectors_3d

# 2D vectors
v1 = Vector(2, 3)
v2 = Vector(-1, 2)
plot_vectors_2d([v1, v2], labels=['v1', 'v2'], title="2D Vectors")

# 3D vectors
i = Vector(1, 0, 0)
j = Vector(0, 1, 0)
k = Vector(0, 0, 1)
plot_vectors_3d([i, j, k], labels=['i', 'j', 'k'], 
                colors=['red', 'green', 'blue'])
```

### Matrix Visualization

```python
from matpy.matrix.core import Matrix
from matpy.visualization import plot_matrix_heatmap, plot_transformation_2d
import math

# Heatmap
m = Matrix(4, 4, [[i+j for j in range(4)] for i in range(4)])
plot_matrix_heatmap(m, title="Matrix Heatmap")

# Transformation visualization
angle = math.pi / 4  # 45 degrees
rotation = Matrix(2, 2, [
    [math.cos(angle), -math.sin(angle)],
    [math.sin(angle), math.cos(angle)]
])
plot_transformation_2d(rotation, title="45° Rotation")
```

---

## Advanced Topics

### Custom Vector Operations

```python
from matpy.vector.core import Vector

# Create custom function using vector operations
def triangle_area(a: Vector, b: Vector, c: Vector) -> float:
    """Calculate area of triangle from 3D points."""
    ab = b - a
    ac = c - a
    cross = ab.cross(ac)
    return cross.magnitude() / 2

# Usage
p1 = Vector(0, 0, 0)
p2 = Vector(1, 0, 0)
p3 = Vector(0, 1, 0)
area = triangle_area(p1, p2, p3)
```

### Matrix Decompositions

```python
from matpy.matrix import solve

# LU decomposition
A = Matrix(3, 3, [[4, 3, 2], [1, 2, 3], [2, 1, 4]])
L, U = solve.lu_decomposition(A)

# Verify: A = L @ U
reconstructed = L @ U
assert reconstructed == A
```

### Performance Tips

```python
# Use @ operator for matrix multiplication (faster)
result = A @ B  # Preferred
result = A * B  # Also works but less clear

# Reuse normalized vectors
unit_v = v.normalize()  # Calculate once
# Use unit_v multiple times

# Use appropriate solver
# Small systems (< 10x10): Cramer's rule
# Medium systems: LU decomposition
# Overdetermined: Least squares
```

---

## Next Steps

- [API Reference](api-reference.md) - Detailed API documentation
- [Examples](examples.md) - More code examples
- [GitHub Repository](https://github.com/njryan-boou/matpy) - Source code

[← Back to Home](index.md)
