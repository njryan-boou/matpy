"""
Matrix visualization functions.

This module provides functions for visualizing matrices and transformations.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ..matrix.core import Matrix
from ..vector.core import Vector


def plot_matrix_heatmap(
    matrix: Matrix,
    title: str = "Matrix Heatmap",
    cmap: str = 'RdBu_r',
    annotate: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot a matrix as a heatmap.
    
    Args:
        matrix: Matrix to visualize
        title: Plot title
        cmap: Colormap name
        annotate: Whether to show values in cells
        ax: Matplotlib axes to plot on (creates new if None)
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to ax.imshow()
    
    Returns:
        The matplotlib axes object
    
    Example:
        >>> m = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> plot_matrix_heatmap(m)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert matrix to numpy array
    data = np.array(matrix.data)
    
    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', **kwargs)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Annotate cells with values
    if annotate:
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                text = ax.text(j, i, f'{matrix.data[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    # Set ticks
    ax.set_xticks(np.arange(matrix.cols))
    ax.set_yticks(np.arange(matrix.rows))
    ax.set_xticklabels(np.arange(matrix.cols))
    ax.set_yticklabels(np.arange(matrix.rows))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    if show:
        plt.show()
    
    return ax


def plot_matrix_grid(
    matrix: Matrix,
    title: str = "Matrix Visualization",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a matrix as a grid with bars representing values.
    
    Args:
        matrix: Matrix to visualize
        title: Plot title
        ax: Matplotlib axes to plot on (creates new if None)
        show: Whether to display the plot immediately
    
    Returns:
        The matplotlib axes object
    
    Example:
        >>> m = Matrix(2, 3, [[1, -2, 3], [4, 5, -6]])
        >>> plot_matrix_grid(m)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart for each row
    x = np.arange(matrix.cols)
    width = 0.8 / matrix.rows
    
    for i in range(matrix.rows):
        offset = (i - matrix.rows/2) * width + width/2
        values = [matrix.data[i][j] for j in range(matrix.cols)]
        ax.bar(x + offset, values, width, label=f'Row {i}')
    
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    if show:
        plt.show()
    
    return ax


def plot_transformation_2d(
    matrix: Matrix,
    vectors: Optional[List[Vector]] = None,
    title: str = "2D Linear Transformation",
    show_basis: bool = True,
    grid_range: Tuple[float, float] = (-3, 3),
    grid_density: int = 7,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Visualize a 2D linear transformation by showing how it transforms vectors and grid.
    
    Args:
        matrix: 2x2 transformation matrix
        vectors: List of vectors to transform (defaults to unit vectors)
        title: Plot title
        show_basis: Whether to show standard basis vectors
        grid_range: Range for background grid
        grid_density: Number of grid lines
        ax: Matplotlib axes to plot on (creates new if None)
        show: Whether to display the plot immediately
    
    Returns:
        The matplotlib axes object
    
    Example:
        >>> # Rotation matrix (90 degrees)
        >>> m = Matrix(2, 2, [[0, -1], [1, 0]])
        >>> plot_transformation_2d(m)
    """
    if matrix.rows != 2 or matrix.cols != 2:
        raise ValueError("Transformation matrix must be 2x2")
    
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        ax_before = ax[0]
        ax_after = ax[1]
    else:
        # If single axis provided, create side-by-side
        fig = plt.figure(figsize=(16, 7))
        ax_before = fig.add_subplot(121)
        ax_after = fig.add_subplot(122)
    
    # Default vectors
    if vectors is None:
        vectors = [Vector(1, 0), Vector(0, 1)]
    
    # Add basis vectors if requested
    if show_basis and vectors != [Vector(1, 0), Vector(0, 1)]:
        vectors = [Vector(1, 0), Vector(0, 1)] + vectors
    
    # Plot grid in background
    grid_vals = np.linspace(grid_range[0], grid_range[1], grid_density)
    for val in grid_vals:
        # Vertical lines
        ax_before.plot([val, val], [grid_range[0], grid_range[1]], 
                      'gray', alpha=0.3, linewidth=0.5)
        # Horizontal lines
        ax_before.plot([grid_range[0], grid_range[1]], [val, val],
                      'gray', alpha=0.3, linewidth=0.5)
    
    # Plot original vectors
    colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))
    for i, (vec, color) in enumerate(zip(vectors, colors)):
        ax_before.arrow(0, 0, vec.x, vec.y, 
                       head_width=0.2, head_length=0.2,
                       fc=color, ec=color, linewidth=2,
                       length_includes_head=True)
        ax_before.text(vec.x, vec.y, f'v{i}', fontsize=12, color=color)
    
    # Transform grid
    for val in grid_vals:
        # Transform vertical line
        v1 = Vector(val, grid_range[0])
        v2 = Vector(val, grid_range[1])
        # Apply transformation
        t1_data = [matrix.data[0][0] * v1.x + matrix.data[0][1] * v1.y,
                   matrix.data[1][0] * v1.x + matrix.data[1][1] * v1.y]
        t2_data = [matrix.data[0][0] * v2.x + matrix.data[0][1] * v2.y,
                   matrix.data[1][0] * v2.x + matrix.data[1][1] * v2.y]
        ax_after.plot([t1_data[0], t2_data[0]], [t1_data[1], t2_data[1]],
                     'gray', alpha=0.3, linewidth=0.5)
        
        # Transform horizontal line
        h1 = Vector(grid_range[0], val)
        h2 = Vector(grid_range[1], val)
        t1_data = [matrix.data[0][0] * h1.x + matrix.data[0][1] * h1.y,
                   matrix.data[1][0] * h1.x + matrix.data[1][1] * h1.y]
        t2_data = [matrix.data[0][0] * h2.x + matrix.data[0][1] * h2.y,
                   matrix.data[1][0] * h2.x + matrix.data[1][1] * h2.y]
        ax_after.plot([t1_data[0], t2_data[0]], [t1_data[1], t2_data[1]],
                     'gray', alpha=0.3, linewidth=0.5)
    
    # Transform and plot vectors
    for i, (vec, color) in enumerate(zip(vectors, colors)):
        # Apply matrix transformation: result = matrix * vector
        transformed_data = [
            matrix.data[0][0] * vec.x + matrix.data[0][1] * vec.y,
            matrix.data[1][0] * vec.x + matrix.data[1][1] * vec.y
        ]
        ax_after.arrow(0, 0, transformed_data[0], transformed_data[1],
                      head_width=0.2, head_length=0.2,
                      fc=color, ec=color, linewidth=2,
                      length_includes_head=True)
        ax_after.text(transformed_data[0], transformed_data[1], 
                     f'Tv{i}', fontsize=12, color=color)
    
    # Configure axes
    for axis in [ax_before, ax_after]:
        axis.set_aspect('equal')
        axis.grid(True, alpha=0.5)
        axis.axhline(y=0, color='k', linewidth=1)
        axis.axvline(x=0, color='k', linewidth=1)
        axis.set_xlim(grid_range)
        axis.set_ylim(grid_range)
    
    ax_before.set_title('Original', fontsize=12, fontweight='bold')
    ax_after.set_title('Transformed', fontsize=12, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if show:
        plt.show()
    
    return ax


def plot_transformation_3d(
    matrix: Matrix,
    vectors: Optional[List[Vector]] = None,
    title: str = "3D Linear Transformation",
    show_basis: bool = True,
    ax: Optional[Tuple[Axes3D, Axes3D]] = None,
    show: bool = True,
) -> Tuple[Axes3D, Axes3D]:
    """
    Visualize a 3D linear transformation.
    
    Args:
        matrix: 3x3 transformation matrix
        vectors: List of 3D vectors to transform (defaults to unit vectors)
        title: Plot title
        show_basis: Whether to show standard basis vectors
        ax: Tuple of two 3D axes (before, after)
        show: Whether to display the plot immediately
    
    Returns:
        Tuple of (before_ax, after_ax)
    
    Example:
        >>> # Rotation around z-axis
        >>> import math
        >>> theta = math.pi / 4
        >>> m = Matrix(3, 3, [
        ...     [math.cos(theta), -math.sin(theta), 0],
        ...     [math.sin(theta), math.cos(theta), 0],
        ...     [0, 0, 1]
        ... ])
        >>> plot_transformation_3d(m)
    """
    if matrix.rows != 3 or matrix.cols != 3:
        raise ValueError("Transformation matrix must be 3x3")
    
    if ax is None:
        fig = plt.figure(figsize=(16, 7))
        ax_before = fig.add_subplot(121, projection='3d')
        ax_after = fig.add_subplot(122, projection='3d')
    else:
        ax_before, ax_after = ax
    
    # Default vectors
    if vectors is None:
        vectors = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
    
    # Add basis vectors if requested
    if show_basis and vectors != [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]:
        vectors = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)] + vectors
    
    # Plot original vectors
    colors = ['red', 'green', 'blue'] + ['purple', 'orange', 'cyan', 'magenta']
    for i, (vec, color) in enumerate(zip(vectors, colors[:len(vectors)])):
        ax_before.quiver(0, 0, 0, vec.x, vec.y, vec.z,
                        color=color, arrow_length_ratio=0.2, linewidth=2)
    
    # Transform and plot vectors
    for i, (vec, color) in enumerate(zip(vectors, colors[:len(vectors)])):
        # Apply matrix transformation
        transformed_data = [
            matrix.data[0][0] * vec.x + matrix.data[0][1] * vec.y + matrix.data[0][2] * vec.z,
            matrix.data[1][0] * vec.x + matrix.data[1][1] * vec.y + matrix.data[1][2] * vec.z,
            matrix.data[2][0] * vec.x + matrix.data[2][1] * vec.y + matrix.data[2][2] * vec.z
        ]
        ax_after.quiver(0, 0, 0, transformed_data[0], transformed_data[1], transformed_data[2],
                       color=color, arrow_length_ratio=0.2, linewidth=2)
    
    # Configure axes
    for axis in [ax_before, ax_after]:
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')
        axis.set_xlim([-2, 2])
        axis.set_ylim([-2, 2])
        axis.set_zlim([-2, 2])
    
    ax_before.set_title('Original', fontsize=12, fontweight='bold')
    ax_after.set_title('Transformed', fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    if show:
        plt.show()
    
    return ax_before, ax_after
