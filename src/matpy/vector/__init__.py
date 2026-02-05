from .core import Vector
from .ops import (
    dot, cross, magnitude, normalize, angle_between, 
    projection, rejection, reflect, lerp,
    distance, distance_squared, component_min, component_max,
    clamp, sum_components, product_components,
    is_parallel, is_perpendicular
)
from .coordinates import (
    VectorCoordinates,
    to_polar, from_polar,
    to_complex, from_complex,
    to_spherical, from_spherical,
    to_cylindrical, from_cylindrical
)

__all__ = [
    'Vector',
    # Basic ops
    'dot', 'cross', 'magnitude', 'normalize', 'angle_between',
    'projection', 'rejection', 'reflect', 'lerp',
    # Additional ops
    'distance', 'distance_squared', 'component_min', 'component_max',
    'clamp', 'sum_components', 'product_components',
    'is_parallel', 'is_perpendicular',
    # Coordinate conversions
    'VectorCoordinates',
    'to_polar', 'from_polar',
    'to_complex', 'from_complex',
    'to_spherical', 'from_spherical',
    'to_cylindrical', 'from_cylindrical'
]