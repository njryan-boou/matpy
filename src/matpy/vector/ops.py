"""
Vector operations module.

This module provides standalone functions for common vector operations.
These functions complement the Vector class methods in core.py.
"""

from __future__ import annotations
import math

from .core import Vector
from ..core import validate, utils


# ==================== Basic Vector Operations ====================

def dot(v1: Vector, v2: Vector) -> float:
    """
    Calculate the dot product of two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        The dot product (scalar value)
    """
    return v1.dot(v2)


def cross(v1: Vector, v2: Vector) -> Vector:
    """
    Calculate the cross product of two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        A new vector perpendicular to both input vectors
    """
    return v1.cross(v2)


def magnitude(v: Vector) -> float:
    """
    Calculate the magnitude (length) of a vector.
    
    Args:
        v: The vector
    
    Returns:
        The magnitude of the vector
    """
    return abs(v)


def normalize(v: Vector) -> Vector:
    """
    Return a normalized (unit) vector in the same direction.
    
    Args:
        v: The vector to normalize
    
    Returns:
        A unit vector in the same direction, or zero vector if input is zero
    """
    return v.normalize()


# ==================== Advanced Vector Operations ====================

def angle_between(v1: Vector, v2: Vector) -> float:
    """
    Calculate the angle between two vectors in radians.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        The angle between the vectors in radians [0, π]
    
    Note:
        Returns 0.0 if either vector has zero magnitude
    """
    dot_prod = v1.dot(v2)
    mag_v1 = abs(v1)
    mag_v2 = abs(v2)
    
    if validate.approx_zero(mag_v1) or validate.approx_zero(mag_v2):
        return 0.0
    
    # Clamp to [-1, 1] to handle floating-point errors
    cos_angle = utils.clamp(dot_prod / (mag_v1 * mag_v2), -1.0, 1.0)
    return math.acos(cos_angle)


def projection(v1: Vector, v2: Vector) -> Vector:
    """
    Project vector v1 onto vector v2.
    
    Args:
        v1: The vector to be projected
        v2: The vector to project onto
    
    Returns:
        The projection of v1 onto v2
    
    Note:
        Returns zero vector if v2 has zero magnitude.
        Vectors must have the same dimension.
    
    Raises:
        DimensionError: If vectors have different dimensions
    """
    validate.validate_dimensions_match(
        len(v1.components),
        len(v2.components),
        "projection"
    )
    
    mag_v2 = abs(v2)
    if validate.approx_zero(mag_v2):
        return Vector(*([0] * len(v1.components)))
    
    scale = v1.dot(v2) / (mag_v2 ** 2)
    return v2 * scale


def rejection(v1: Vector, v2: Vector) -> Vector:
    """
    Calculate the rejection of v1 from v2.
    
    The rejection is the component of v1 perpendicular to v2.
    
    Args:
        v1: The vector to be rejected
        v2: The vector to reject from
    
    Returns:
        The component of v1 perpendicular to v2
    
    Note:
        projection(v1, v2) + rejection(v1, v2) = v1
    """
    return v1 - projection(v1, v2)


def reflect(v: Vector, normal: Vector) -> Vector:
    """
    Reflect a vector across a surface defined by its normal.
    
    Args:
        v: The vector to reflect
        normal: The surface normal vector
    
    Returns:
        The reflected vector
    
    Note:
        The normal vector is automatically normalized
    """
    normal = normal.normalize()
    return v - 2 * v.dot(normal) * normal


def lerp(v1: Vector, v2: Vector, t: float) -> Vector:
    """
    Linear interpolation between two vectors.
    
    Args:
        v1: Start vector (when t=0)
        v2: End vector (when t=1)
        t: Interpolation parameter
    
    Returns:
        Interpolated vector
    
    Note:
        - t=0 returns v1
        - t=1 returns v2
        - t=0.5 returns the midpoint
        - t can be outside [0, 1] for extrapolation
    
    Raises:
        DimensionError: If vectors have different dimensions
    """
    validate.validate_dimensions_match(
        len(v1.components),
        len(v2.components),
        "linear interpolation"
    )
    return v1 * (1 - t) + v2 * t


def distance(v1: Vector, v2: Vector) -> float:
    """
    Calculate the Euclidean distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        The distance between the vectors
    
    Raises:
        DimensionError: If vectors have different dimensions
    """
    validate.validate_dimensions_match(
        len(v1.components),
        len(v2.components),
        "distance calculation"
    )
    return abs(v2 - v1)


def distance_squared(v1: Vector, v2: Vector) -> float:
    """
    Calculate the squared Euclidean distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        The squared distance (avoids sqrt for performance)
    
    Raises:
        DimensionError: If vectors have different dimensions
    """
    validate.validate_dimensions_match(
        len(v1.components),
        len(v2.components),
        "distance squared calculation"
    )
    diff = v2 - v1
    return sum(c ** 2 for c in diff.components)


def component_min(v1: Vector, v2: Vector) -> Vector:
    """
    Create a vector with the minimum components from two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Vector with min(v1[i], v2[i]) for each component i
    
    Raises:
        DimensionError: If vectors have different dimensions
    """
    validate.validate_dimensions_match(
        len(v1.components),
        len(v2.components),
        "component-wise minimum"
    )
    return Vector(*(min(a, b) for a, b in zip(v1.components, v2.components)))


def component_max(v1: Vector, v2: Vector) -> Vector:
    """
    Create a vector with the maximum components from two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Vector with max(v1[i], v2[i]) for each component i
    
    Raises:
        DimensionError: If vectors have different dimensions
    """
    validate.validate_dimensions_match(
        len(v1.components),
        len(v2.components),
        "component-wise maximum"
    )
    return Vector(*(max(a, b) for a, b in zip(v1.components, v2.components)))


def clamp(v: Vector, min_val: float, max_val: float) -> Vector:
    """
    Clamp all vector components to a range.
    
    Args:
        v: The vector to clamp
        min_val: Minimum value for each component
        max_val: Maximum value for each component
    
    Returns:
        Vector with all components clamped to [min_val, max_val]
    """
    return Vector(*(max(min_val, min(max_val, c)) for c in v.components))


def sum_components(v: Vector) -> float:
    """
    Calculate the sum of all vector components.
    
    Args:
        v: The vector
    
    Returns:
        Sum of all components
    """
    return sum(v.components)


def product_components(v: Vector) -> float:
    """
    Calculate the product of all vector components.
    
    Args:
        v: The vector
    
    Returns:
        Product of all components
    """
    result = 1.0
    for c in v.components:
        result *= c
    return result


def is_parallel(v1: Vector, v2: Vector, tolerance: float = 1e-10) -> bool:
    """
    Check if two vectors are parallel.
    
    Args:
        v1: First vector
        v2: Second vector
        tolerance: Tolerance for comparison (default 1e-10)
    
    Returns:
        True if vectors are parallel, False otherwise
    
    Note:
        Two vectors are parallel if one is a scalar multiple of the other.
        Zero vectors are considered parallel to everything.
    """
    mag1 = abs(v1)
    mag2 = abs(v2)
    
    if validate.approx_zero(mag1, tolerance) or validate.approx_zero(mag2, tolerance):
        return True
    
    # Normalize and check if dot product is ±1
    normalized_dot = abs(v1.dot(v2) / (mag1 * mag2))
    return validate.approx_equal(normalized_dot, 1.0, tolerance)


def is_perpendicular(v1: Vector, v2: Vector, tolerance: float = 1e-10) -> bool:
    """
    Check if two vectors are perpendicular (orthogonal).
    
    Args:
        v1: First vector
        v2: Second vector
        tolerance: Tolerance for comparison (default 1e-10)
    
    Returns:
        True if vectors are perpendicular, False otherwise
    
    Note:
        Two vectors are perpendicular if their dot product is zero.
    """
    return validate.approx_zero(v1.dot(v2), tolerance)

