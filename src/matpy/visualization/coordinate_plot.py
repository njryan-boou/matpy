"""
Coordinate system visualization functions.

This module provides functions for visualizing different coordinate systems.
"""

from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ..vector.core import Vector
from ..vector.coordinates import VectorCoordinates


def plot_coordinate_systems_2d(
    vector: Vector,
    title: str = "2D Coordinate Systems",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Visualize a 2D vector in Cartesian and Polar coordinate systems.
    
    Args:
        vector: 2D vector to visualize
        title: Plot title
        ax: Matplotlib axes to plot on (creates new if None)
        show: Whether to display the plot immediately
    
    Returns:
        The matplotlib axes object
    
    Example:
        >>> v = Vector(3, 4)
        >>> plot_coordinate_systems_2d(v)
    """
    if len(vector.components) != 2:
        raise ValueError("Vector must be 2D")
    
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig = plt.gcf()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122, projection='polar')
    
    coords = VectorCoordinates(vector)
    r, theta = coords.to_polar()
    
    # Cartesian plot
    ax1.arrow(0, 0, vector.x, vector.y,
             head_width=0.2, head_length=0.3,
             fc='blue', ec='blue', linewidth=2,
             length_includes_head=True)
    ax1.plot([0, vector.x], [0, 0], 'r--', alpha=0.5, label=f'x = {vector.x:.2f}')
    ax1.plot([vector.x, vector.x], [0, vector.y], 'g--', alpha=0.5, label=f'y = {vector.y:.2f}')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_title('Cartesian (x, y)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Polar plot
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot([0, theta], [0, r], 'b-', linewidth=2, marker='o')
    ax2.set_title(f'Polar (r={r:.2f}, θ={theta:.2f} rad)', 
                 fontsize=12, fontweight='bold', pad=20)
    ax2.grid(True)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if show:
        plt.show()
    
    return ax1


def plot_coordinate_systems_3d(
    vector: Vector,
    title: str = "3D Coordinate Systems",
    show: bool = True,
) -> None:
    """
    Visualize a 3D vector in Cartesian, Spherical, and Cylindrical coordinate systems.
    
    Args:
        vector: 3D vector to visualize
        title: Plot title
        show: Whether to display the plot immediately
    
    Example:
        >>> v = Vector(1, 1, 1)
        >>> plot_coordinate_systems_3d(v)
    """
    if len(vector.components) != 3:
        raise ValueError("Vector must be 3D")
    
    coords = VectorCoordinates(vector)
    r_sph, theta_sph, phi_sph = coords.to_spherical()
    rho_cyl, phi_cyl, z_cyl = coords.to_cylindrical()
    
    fig = plt.figure(figsize=(18, 6))
    
    # Cartesian
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.quiver(0, 0, 0, vector.x, vector.y, vector.z,
              color='blue', arrow_length_ratio=0.2, linewidth=2)
    ax1.plot([0, vector.x], [0, 0], [0, 0], 'r--', alpha=0.5)
    ax1.plot([vector.x, vector.x], [0, vector.y], [0, 0], 'g--', alpha=0.5)
    ax1.plot([vector.x, vector.x], [vector.y, vector.y], [0, vector.z], 'b--', alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Cartesian\n(x={vector.x:.2f}, y={vector.y:.2f}, z={vector.z:.2f})',
                 fontsize=10, fontweight='bold')
    
    # Spherical
    ax2 = fig.add_subplot(132, projection='3d')
    # Draw vector
    ax2.quiver(0, 0, 0, vector.x, vector.y, vector.z,
              color='blue', arrow_length_ratio=0.2, linewidth=2)
    # Draw radius from origin to point
    ax2.plot([0, vector.x], [0, vector.y], [0, vector.z], 'r-', linewidth=1, alpha=0.7)
    # Draw projection on xy-plane
    ax2.plot([0, vector.x], [0, vector.y], [0, 0], 'g--', alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Spherical\n(r={r_sph:.2f}, θ={theta_sph:.2f}, φ={phi_sph:.2f})',
                 fontsize=10, fontweight='bold')
    
    # Cylindrical
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.quiver(0, 0, 0, vector.x, vector.y, vector.z,
              color='blue', arrow_length_ratio=0.2, linewidth=2)
    # Draw rho (radius in xy-plane)
    ax3.plot([0, vector.x], [0, vector.y], [0, 0], 'r-', linewidth=2, alpha=0.7)
    # Draw z
    ax3.plot([vector.x, vector.x], [vector.y, vector.y], [0, vector.z], 'g-', linewidth=2, alpha=0.7)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Cylindrical\n(ρ={rho_cyl:.2f}, φ={phi_cyl:.2f}, z={z_cyl:.2f})',
                 fontsize=10, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if show:
        plt.show()
