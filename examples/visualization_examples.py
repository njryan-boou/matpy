"""
Visualization examples for matpy library.

This module demonstrates various visualization capabilities for vectors,
matrices, and transformations.

Note: Requires matplotlib. Install with: pip install matpy-linalg[viz]
"""

import math
from matpy.vector.core import Vector
from matpy.matrix.core import Matrix
from matpy.visualization import (
    plot_vector_2d,
    plot_vectors_2d,
    plot_vector_3d,
    plot_vectors_3d,
    plot_vector_field_2d,
    plot_matrix_heatmap,
    plot_matrix_grid,
    plot_transformation_2d,
    plot_transformation_3d,
    plot_coordinate_systems_2d,
    plot_coordinate_systems_3d,
)


def example_vector_2d():
    """Example: Plot 2D vectors."""
    print("=== 2D Vector Plotting ===")
    
    # Single vector
    v = Vector(3, 4)
    plot_vector_2d(v, label='v = (3, 4)', color='blue')
    
    # Multiple vectors
    v1 = Vector(2, 3)
    v2 = Vector(-1, 2)
    v3 = Vector(3, -1)
    plot_vectors_2d(
        [v1, v2, v3],
        labels=['v1', 'v2', 'v3'],
        title="Multiple 2D Vectors"
    )


def example_vector_3d():
    """Example: Plot 3D vectors."""
    print("=== 3D Vector Plotting ===")
    
    # Standard basis vectors
    i = Vector(1, 0, 0)
    j = Vector(0, 1, 0)
    k = Vector(0, 0, 1)
    
    plot_vectors_3d(
        [i, j, k],
        labels=['i', 'j', 'k'],
        colors=['red', 'green', 'blue'],
        title="Standard Basis Vectors"
    )
    
    # Custom vectors
    v1 = Vector(1, 2, 2)
    v2 = Vector(-1, 1, 2)
    plot_vectors_3d([v1, v2], labels=['v1', 'v2'])


def example_vector_field():
    """Example: Plot 2D vector field."""
    print("=== Vector Field Plotting ===")
    
    # Rotation field
    def rotation_field(x, y):
        return Vector(-y, x)
    
    plot_vector_field_2d(
        rotation_field,
        x_range=(-3, 3),
        y_range=(-3, 3),
        density=15,
        title="Rotation Field: (-y, x)"
    )
    
    # Gradient field
    def gradient_field(x, y):
        return Vector(x, y)
    
    plot_vector_field_2d(
        gradient_field,
        title="Gradient Field: (x, y)"
    )


def example_matrix_heatmap():
    """Example: Matrix heatmap visualization."""
    print("=== Matrix Heatmap ===")
    
    # Create a sample matrix
    m = Matrix(4, 4, [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    
    plot_matrix_heatmap(m, title="4x4 Matrix Heatmap")
    
    # Symmetric matrix
    symmetric = Matrix(3, 3, [
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ])
    plot_matrix_heatmap(symmetric, title="Symmetric Matrix", cmap='viridis')


def example_matrix_grid():
    """Example: Matrix grid visualization."""
    print("=== Matrix Grid ===")
    
    m = Matrix(3, 5, [
        [1, -2, 3, 4, -1],
        [5, 6, -3, 2, 4],
        [-2, 3, 5, -4, 1]
    ])
    
    plot_matrix_grid(m, title="Matrix as Bar Chart")


def example_transformation_2d():
    """Example: 2D linear transformation visualization."""
    print("=== 2D Transformations ===")
    
    # Rotation by 45 degrees
    angle = math.pi / 4
    rotation = Matrix(2, 2, [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])
    
    plot_transformation_2d(
        rotation,
        title="Rotation by 45°"
    )
    
    # Scaling
    scaling = Matrix(2, 2, [
        [2, 0],
        [0, 0.5]
    ])
    
    plot_transformation_2d(
        scaling,
        vectors=[Vector(1, 1), Vector(-1, 1)],
        title="Scaling Transformation (2x in x, 0.5x in y)"
    )
    
    # Shear
    shear = Matrix(2, 2, [
        [1, 1],
        [0, 1]
    ])
    
    plot_transformation_2d(
        shear,
        title="Shear Transformation"
    )
    
    # Reflection across y=x
    reflection = Matrix(2, 2, [
        [0, 1],
        [1, 0]
    ])
    
    plot_transformation_2d(
        reflection,
        title="Reflection across y=x"
    )


def example_transformation_3d():
    """Example: 3D linear transformation visualization."""
    print("=== 3D Transformations ===")
    
    # Rotation around z-axis
    angle = math.pi / 3
    rotation_z = Matrix(3, 3, [
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
    
    plot_transformation_3d(
        rotation_z,
        title="Rotation around Z-axis by 60°"
    )
    
    # Scaling
    scaling_3d = Matrix(3, 3, [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 0.5]
    ])
    
    plot_transformation_3d(
        scaling_3d,
        vectors=[Vector(1, 1, 1)],
        title="3D Scaling"
    )


def example_coordinate_systems_2d():
    """Example: 2D coordinate system visualization."""
    print("=== 2D Coordinate Systems ===")
    
    v = Vector(3, 4)
    plot_coordinate_systems_2d(v, title="Vector in Cartesian and Polar Coordinates")
    
    v2 = Vector(-2, 3)
    plot_coordinate_systems_2d(v2, title="Another Vector")


def example_coordinate_systems_3d():
    """Example: 3D coordinate system visualization."""
    print("=== 3D Coordinate Systems ===")
    
    v = Vector(2, 2, 3)
    plot_coordinate_systems_3d(v, title="Vector in 3D Coordinate Systems")
    
    v2 = Vector(1, 1, 1)
    plot_coordinate_systems_3d(v2, title="Unit Diagonal Vector")


def example_vector_operations():
    """Example: Visualize vector operations."""
    print("=== Vector Operations ===")
    
    import matplotlib.pyplot as plt
    from matpy.vector import ops
    
    # Vector addition
    v1 = Vector(2, 1)
    v2 = Vector(1, 3)
    v_sum = v1 + v2
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_vector_2d(v1, label='v1', color='red', ax=ax, show=False)
    plot_vector_2d(v2, origin=(v1.x, v1.y), label='v2', color='blue', ax=ax, show=False)
    plot_vector_2d(v_sum, label='v1 + v2', color='green', ax=ax, show=False)
    ax.set_title('Vector Addition', fontsize=14, fontweight='bold')
    plt.show()
    
    # Vector projection
    v1 = Vector(3, 2)
    v2 = Vector(4, 0)
    proj = ops.projection(v1, v2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_vector_2d(v1, label='v1', color='red', ax=ax, show=False)
    plot_vector_2d(v2, label='v2', color='blue', ax=ax, show=False)
    plot_vector_2d(proj, label='proj(v1, v2)', color='green', ax=ax, show=False)
    ax.set_title('Vector Projection', fontsize=14, fontweight='bold')
    plt.show()


def main():
    """Run all visualization examples."""
    print("MatPy Visualization Examples")
    print("=" * 50)
    
    # Comment out examples you don't want to run
    example_vector_2d()
    example_vector_3d()
    example_vector_field()
    example_matrix_heatmap()
    example_matrix_grid()
    example_transformation_2d()
    example_transformation_3d()
    example_coordinate_systems_2d()
    example_coordinate_systems_3d()
    example_vector_operations()
    
    print("\nAll examples complete!")


if __name__ == "__main__":
    # Check if matplotlib is installed
    try:
        import matplotlib
        main()
    except ImportError:
        print("Matplotlib is required for visualization examples.")
        print("Install with: pip install matpy-linalg[viz]")
