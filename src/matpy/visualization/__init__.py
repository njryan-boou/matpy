"""
Visualization module for matpy library.

This module provides visualization tools for vectors, matrices, and transformations
using matplotlib.

Note: This module requires matplotlib to be installed.
Install with: pip install matpy-linalg[viz]
"""

from __future__ import annotations

# Check if matplotlib is available
try:
    import matplotlib
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

if _HAS_MATPLOTLIB:
    from .vector_plot import (
        plot_vector_2d,
        plot_vectors_2d,
        plot_vector_3d,
        plot_vectors_3d,
        plot_vector_field_2d,
    )
    from .matrix_plot import (
        plot_matrix_heatmap,
        plot_matrix_grid,
        plot_transformation_2d,
        plot_transformation_3d,
    )
    from .coordinate_plot import (
        plot_coordinate_systems_2d,
        plot_coordinate_systems_3d,
    )

    __all__ = [
        # Vector plotting
        'plot_vector_2d',
        'plot_vectors_2d',
        'plot_vector_3d',
        'plot_vectors_3d',
        'plot_vector_field_2d',
        # Matrix plotting
        'plot_matrix_heatmap',
        'plot_matrix_grid',
        'plot_transformation_2d',
        'plot_transformation_3d',
        # Coordinate plotting
        'plot_coordinate_systems_2d',
        'plot_coordinate_systems_3d',
    ]
else:
    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "Visualization features require matplotlib. "
            "Install with: pip install matpy-linalg[viz]"
        )
    
    # Create placeholder functions that raise import error
    plot_vector_2d = _raise_import_error
    plot_vectors_2d = _raise_import_error
    plot_vector_3d = _raise_import_error
    plot_vectors_3d = _raise_import_error
    plot_vector_field_2d = _raise_import_error
    plot_matrix_heatmap = _raise_import_error
    plot_matrix_grid = _raise_import_error
    plot_transformation_2d = _raise_import_error
    plot_transformation_3d = _raise_import_error
    plot_coordinate_systems_2d = _raise_import_error
    plot_coordinate_systems_3d = _raise_import_error
    
    __all__ = []
