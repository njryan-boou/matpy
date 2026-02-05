"""
Vector visualization functions.

This module provides functions for plotting vectors in 2D and 3D space.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ..vector.core import Vector
from ..core.validate import validate_vector_dimension


def plot_vector_2d(
    vector: Vector,
    origin: Tuple[float, float] = (0, 0),
    color: str = 'blue',
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot a 2D vector as an arrow.
    
    Args:
        vector: 2D Vector to plot
        origin: Starting point (x, y) for the vector
        color: Arrow color
        label: Label for the vector
        ax: Matplotlib axes to plot on (creates new if None)
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to ax.arrow()
    
    Returns:
        The matplotlib axes object
    
    Raises:
        DimensionError: If vector is not 2D
    
    Example:
        >>> v = Vector(3, 4)
        >>> plot_vector_2d(v, label='v', color='red')
    """
    validate_vector_dimension(len(vector.components), 2, "2D vector plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Default arrow properties
    arrow_props = {
        'head_width': 0.2,
        'head_length': 0.3,
        'fc': color,
        'ec': color,
        'length_includes_head': True,
        'width': 0.05,
    }
    arrow_props.update(kwargs)
    
    # Plot arrow
    ax.arrow(
        origin[0], origin[1],
        vector.x, vector.y,
        **arrow_props
    )
    
    # Add label if provided
    if label:
        mid_x = origin[0] + vector.x / 2
        mid_y = origin[1] + vector.y / 2
        ax.text(mid_x, mid_y, label, fontsize=12, color=color)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    if show:
        plt.show()
    
    return ax


def plot_vectors_2d(
    vectors: List[Vector],
    origins: Optional[List[Tuple[float, float]]] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    title: str = "2D Vectors",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot multiple 2D vectors on the same axes.
    
    Args:
        vectors: List of 2D vectors to plot
        origins: List of starting points (defaults to origin for all)
        colors: List of colors for each vector
        labels: List of labels for each vector
        title: Plot title
        ax: Matplotlib axes to plot on (creates new if None)
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to arrow plotting
    
    Returns:
        The matplotlib axes object
    
    Example:
        >>> v1 = Vector(2, 3)
        >>> v2 = Vector(-1, 2)
        >>> plot_vectors_2d([v1, v2], labels=['v1', 'v2'])
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set defaults
    if origins is None:
        origins = [(0, 0)] * len(vectors)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))
    if labels is None:
        labels = [None] * len(vectors)
    
    # Plot each vector
    for vector, origin, color, label in zip(vectors, origins, colors, labels):
        plot_vector_2d(vector, origin, color, label, ax=ax, show=False, **kwargs)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    if show:
        plt.show()
    
    return ax


def plot_vector_3d(
    vector: Vector,
    origin: Tuple[float, float, float] = (0, 0, 0),
    color: str = 'blue',
    label: Optional[str] = None,
    ax: Optional[Axes3D] = None,
    show: bool = True,
    **kwargs
) -> Axes3D:
    """
    Plot a 3D vector as an arrow.
    
    Args:
        vector: 3D Vector to plot
        origin: Starting point (x, y, z) for the vector
        color: Arrow color
        label: Label for the vector
        ax: Matplotlib 3D axes to plot on (creates new if None)
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to ax.quiver()
    
    Returns:
        The matplotlib 3D axes object
    
    Raises:
        DimensionError: If vector is not 3D
    
    Example:
        >>> v = Vector(1, 2, 3)
        >>> plot_vector_3d(v, label='v', color='green')
    """
    validate_vector_dimension(len(vector.components), 3, "3D vector plotting")
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot arrow using quiver
    ax.quiver(
        origin[0], origin[1], origin[2],
        vector.x, vector.y, vector.z,
        color=color,
        arrow_length_ratio=0.2,
        linewidth=2,
        **kwargs
    )
    
    # Add label if provided
    if label:
        mid_x = origin[0] + vector.x / 2
        mid_y = origin[1] + vector.y / 2
        mid_z = origin[2] + vector.z / 2
        ax.text(mid_x, mid_y, mid_z, label, fontsize=12, color=color)
    
    # Set labels and grid
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return ax


def plot_vectors_3d(
    vectors: List[Vector],
    origins: Optional[List[Tuple[float, float, float]]] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    title: str = "3D Vectors",
    ax: Optional[Axes3D] = None,
    show: bool = True,
    **kwargs
) -> Axes3D:
    """
    Plot multiple 3D vectors on the same axes.
    
    Args:
        vectors: List of 3D vectors to plot
        origins: List of starting points (defaults to origin for all)
        colors: List of colors for each vector
        labels: List of labels for each vector
        title: Plot title
        ax: Matplotlib 3D axes to plot on (creates new if None)
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to quiver plotting
    
    Returns:
        The matplotlib 3D axes object
    
    Example:
        >>> v1 = Vector(1, 0, 0)
        >>> v2 = Vector(0, 1, 0)
        >>> v3 = Vector(0, 0, 1)
        >>> plot_vectors_3d([v1, v2, v3], labels=['i', 'j', 'k'])
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Set defaults
    if origins is None:
        origins = [(0, 0, 0)] * len(vectors)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))
    if labels is None:
        labels = [None] * len(vectors)
    
    # Plot each vector
    for vector, origin, color, label in zip(vectors, origins, colors, labels):
        plot_vector_3d(vector, origin, color, label, ax=ax, show=False, **kwargs)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if show:
        plt.show()
    
    return ax


def plot_vector_field_2d(
    func,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    density: int = 20,
    title: str = "2D Vector Field",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot a 2D vector field.
    
    Args:
        func: Function that takes (x, y) and returns a Vector
        x_range: (min, max) range for x-axis
        y_range: (min, max) range for y-axis
        density: Number of arrows in each direction
        title: Plot title
        ax: Matplotlib axes to plot on (creates new if None)
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to ax.quiver()
    
    Returns:
        The matplotlib axes object
    
    Example:
        >>> def f(x, y):
        ...     return Vector(-y, x)  # Rotation field
        >>> plot_vector_field_2d(f)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    # Calculate vector field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(density):
        for j in range(density):
            vec = func(X[i, j], Y[i, j])
            U[i, j] = vec.x
            V[i, j] = vec.y
    
    # Plot vector field
    ax.quiver(X, Y, U, V, **kwargs)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return ax
