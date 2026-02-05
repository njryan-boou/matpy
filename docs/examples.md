---
layout: default
title: Examples
---

# Examples

[← Back to Home](index.md)

Practical examples demonstrating MatPy features.

## Vector Examples

### Basic Vector Arithmetic

```python
from matpy.vector.core import Vector

# Create vectors
v1 = Vector(3, 4)
v2 = Vector(1, 2)

# Operations
print(f"v1 + v2 = {v1 + v2}")      # (4, 6)
print(f"v1 - v2 = {v1 - v2}")      # (2, 2)
print(f"v1 * 3 = {v1 * 3}")        # (9, 12)
print(f"|v1| = {v1.magnitude()}")  # 5.0
```

### 3D Vector Cross Product

```python
v1 = Vector(1, 0, 0)
v2 = Vector(0, 1, 0)
cross = v1.cross(v2)
print(f"v1 × v2 = {cross}")  # (0, 0, 1)
```

### Find Angle Between Vectors

```python
from matpy.vector import ops
import math

v1 = Vector(1, 0)
v2 = Vector(1, 1)
angle_rad = ops.angle_between(v1, v2)
angle_deg = math.degrees(angle_rad)
print(f"Angle: {angle_deg}°")  # 45°
```

## Matrix Examples

### Matrix Multiplication

```python
from matpy.matrix.core import Matrix

A = Matrix(2, 2, [[1, 2], [3, 4]])
B = Matrix(2, 2, [[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B
print(f"A × B = {C}")
# [[19, 22], [43, 50]]
```

### Matrix Inverse

```python
A = Matrix(2, 2, [[4, 7], [2, 6]])
A_inv = A.inverse()

# Verify A × A^(-1) = I
I = A @ A_inv
print(f"A × A^(-1) = {I}")
# [[1, 0], [0, 1]]
```

### Determinant Calculation

```python
A = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 10]])
det = A.determinant()
print(f"det(A) = {det}")  # -3
```

## Linear Systems

### Solve System of Equations

```python
from matpy.matrix.core import Matrix
from matpy.matrix import solve

# System: 2x + y - z = 8
#        -3x - y + 2z = -11
#        -2x + y + 2z = -3

A = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = [8, -11, -3]

x = solve.solve_linear_system(A, b)
print(f"Solution: x={x[0]}, y={x[1]}, z={x[2]}")
# Solution: x=2, y=3, z=-1
```

### Least Squares Fitting

```python
# Fit line y = mx + c to data points
data_x = [1, 2, 3, 4]
data_y = [2, 3, 5, 4]

# Build system Ax = b where x = [m, c]
A = Matrix(4, 2, [[x, 1] for x in data_x])
b = data_y

solution = solve.solve_least_squares(A, b)
m, c = solution
print(f"Best fit line: y = {m:.2f}x + {c:.2f}")
```

## Coordinate Systems

### Polar Coordinates

```python
from matpy.vector.core import Vector
from matpy.vector.coordinates import VectorCoordinates
import math

# Cartesian to Polar
v = Vector(3, 4)
coords = VectorCoordinates(v)
r, theta = coords.to_polar()
print(f"Polar: r={r}, θ={math.degrees(theta)}°")
# Polar: r=5.0, θ=53.13°

# Polar to Cartesian
v2 = VectorCoordinates.from_polar(5, math.radians(53.13))
print(f"Cartesian: {v2}")  # (3, 4)
```

### 3D Spherical Coordinates

```python
v = Vector(1, 1, math.sqrt(2))
coords = VectorCoordinates(v)

# To spherical
r, theta, phi = coords.to_spherical()
print(f"Spherical: r={r:.2f}, θ={math.degrees(theta):.1f}°, φ={math.degrees(phi):.1f}°")

# From spherical
v2 = VectorCoordinates.from_spherical(2, math.pi/4, math.pi/4)
print(f"Cartesian: {v2}")
```

## Differential Equations

### Solve ODE System

```python
from matpy.matrix.core import Matrix
from matpy.matrix import solve

# Solve dx/dt = Ax with A = [[-1, 1], [0, -2]]
# Initial condition: x(0) = [1, 0]

A = Matrix(2, 2, [[-1, 1], [0, -2]])
x0 = [1, 0]
t = 1.0

x_t = solve.solve_linear_ode_system_homogeneous(A, x0, t)
print(f"x({t}) = {x_t}")
```

### Population Model

```python
# Leslie matrix model for population dynamics
# Age groups: juveniles, adults

# Survival and reproduction rates
survival_juvenile = 0.5
survival_adult = 0.9
birth_rate = 1.5

# Leslie matrix
L = Matrix(2, 2, [
    [0, birth_rate],
    [survival_juvenile, survival_adult]
])

# Initial population
pop = [100, 50]  # [juveniles, adults]

# Project 5 years
for year in range(5):
    pop_vector = Vector(*pop)
    # Convert to matrix operation
    pop_matrix = Matrix(2, 1, [[pop[0]], [pop[1]]])
    result = L @ pop_matrix
    pop = [result.data[0][0], result.data[1][0]]
    print(f"Year {year+1}: Juv={pop[0]:.0f}, Adult={pop[1]:.0f}")
```

## Visualization Examples

### Plot Vectors

```python
from matpy.vector.core import Vector
from matpy.visualization import plot_vectors_2d
import matplotlib.pyplot as plt

# Create vectors
vectors = [
    Vector(2, 3),
    Vector(-1, 2),
    Vector(1, -1)
]

labels = ['v1', 'v2', 'v3']
colors = ['red', 'blue', 'green']

plot_vectors_2d(vectors, labels=labels, colors=colors, title="My Vectors")
plt.show()
```

### Rotation Matrix Visualization

```python
from matpy.matrix.core import Matrix
from matpy.visualization import plot_transformation_2d
import math

# 45-degree rotation
angle = math.pi / 4
rotation = Matrix(2, 2, [
    [math.cos(angle), -math.sin(angle)],
    [math.sin(angle), math.cos(angle)]
])

plot_transformation_2d(rotation, title="45° Rotation")
plt.show()
```

### Matrix Heatmap

```python
from matpy.matrix.core import Matrix
from matpy.visualization import plot_matrix_heatmap

# Create correlation matrix
m = Matrix(4, 4, [
    [1.0, 0.8, 0.3, 0.1],
    [0.8, 1.0, 0.5, 0.2],
    [0.3, 0.5, 1.0, 0.7],
    [0.1, 0.2, 0.7, 1.0]
])

plot_matrix_heatmap(m, title="Correlation Matrix", cmap='coolwarm')
plt.show()
```

## Advanced Examples

### Gram-Schmidt Orthogonalization

```python
from matpy.vector.core import Vector
from matpy.vector import ops

def gram_schmidt(vectors):
    """Orthogonalize vectors using Gram-Schmidt process."""
    orthogonal = []
    
    for v in vectors:
        # Subtract projections onto previous vectors
        u = v
        for basis in orthogonal:
            u = u - ops.projection(v, basis)
        
        # Normalize
        orthogonal.append(u.normalize())
    
    return orthogonal

# Test with 3D vectors
v1 = Vector(1, 1, 0)
v2 = Vector(1, 0, 1)
v3 = Vector(0, 1, 1)

orthonormal = gram_schmidt([v1, v2, v3])
for i, v in enumerate(orthonormal):
    print(f"e{i+1} = {v}")
```

### Eigenvalue Approximation (Power Method)

```python
from matpy.matrix.core import Matrix
from matpy.vector.core import Vector

def power_method(A, iterations=100):
    """Approximate dominant eigenvalue using power method."""
    # Start with random vector
    v = Vector(*[1.0] * A.rows)
    
    for _ in range(iterations):
        # Convert vector to matrix for multiplication
        v_matrix = Matrix(A.rows, 1, [[c] for c in v.components])
        Av = A @ v_matrix
        
        # Extract result as vector
        v_new = Vector(*[Av.data[i][0] for i in range(A.rows)])
        v = v_new.normalize()
    
    # Calculate eigenvalue
    v_matrix = Matrix(A.rows, 1, [[c] for c in v.components])
    Av = A @ v_matrix
    lambda_val = sum(Av.data[i][0] * v.components[i] for i in range(A.rows))
    
    return lambda_val, v

# Test
A = Matrix(2, 2, [[4, 1], [1, 3]])
eigenvalue, eigenvector = power_method(A)
print(f"Dominant eigenvalue: {eigenvalue:.4f}")
print(f"Eigenvector: {eigenvector}")
```

### Solve Heat Equation

```python
from matpy.matrix.core import Matrix
from matpy.matrix import solve
import math

# 1D heat equation: du/dt = k * d²u/dx²
# Discretized using finite differences

n = 5  # Grid points
dx = 1.0 / n
k = 0.1  # Thermal diffusivity
dt = 0.01

# Build finite difference matrix
A_data = [[0.0] * n for _ in range(n)]
for i in range(n):
    if i > 0:
        A_data[i][i-1] = k / (dx * dx)
    A_data[i][i] = -2 * k / (dx * dx)
    if i < n - 1:
        A_data[i][i+1] = k / (dx * dx)

A = Matrix(n, n, A_data)

# Initial temperature distribution
u0 = [math.sin(math.pi * i * dx) for i in range(n)]

# Solve for temperature at t=0.1
u_t = solve.solve_linear_ode_system_homogeneous(A, u0, 0.1)
print(f"Temperature at t=0.1: {u_t}")
```

---

## More Examples

Check out the `examples/` directory in the repository:

- `vector_arithmatic.py` - Vector operations
- `matrix_examples.py` - Complete matrix showcase
- `visualization_examples.py` - All visualization features
- `python_methods.py` - Python protocol demonstrations

[← Back to Home](index.md)
