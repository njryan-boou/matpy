"""
Validation utilities for matpy library.

This module provides validation functions used throughout the library
for checking inputs, dimensions, types, and mathematical constraints.
"""

from __future__ import annotations
from typing import Any, List, Tuple, Union
import numbers

from ..error import (
    ValidationError, DimensionError, ShapeError, 
    NotSquareError, ZeroMagnitudeError
)


# ==================== Type Validation ====================

def validate_numeric(value: Any, name: str = "value") -> None:
    """
    Validate that a value is numeric (int or float).
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
    
    Raises:
        ValidationError: If value is not numeric
    
    Example:
        >>> validate_numeric(3.14, "radius")
        >>> validate_numeric("hello", "radius")  # Raises ValidationError
    """
    if not isinstance(value, numbers.Number):
        raise ValidationError(
            f"{name} must be numeric, got {type(value).__name__}"
        )


def validate_integer(value: Any, name: str = "value") -> None:
    """
    Validate that a value is an integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
    
    Raises:
        ValidationError: If value is not an integer
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        )


def validate_positive(value: float, name: str = "value", strict: bool = False) -> None:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        strict: If True, value must be > 0; if False, value must be >= 0
    
    Raises:
        ValidationError: If value is not positive
    
    Example:
        >>> validate_positive(5, "count")
        >>> validate_positive(0, "count", strict=True)  # Raises ValidationError
    """
    if strict:
        if value <= 0:
            raise ValidationError(f"{name} must be strictly positive, got {value}")
    else:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_range(value: float, min_val: float, max_val: float, 
                   name: str = "value", inclusive: bool = True) -> None:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter for error messages
        inclusive: If True, endpoints are included; if False, excluded
    
    Raises:
        ValidationError: If value is outside range
    
    Example:
        >>> validate_range(0.5, 0, 1, "probability")
        >>> validate_range(2, 0, 1, "probability")  # Raises ValidationError
    """
    if inclusive:
        if not (min_val <= value <= max_val):
            raise ValidationError(
                f"{name} must be in range [{min_val}, {max_val}], got {value}"
            )
    else:
        if not (min_val < value < max_val):
            raise ValidationError(
                f"{name} must be in range ({min_val}, {max_val}), got {value}"
            )


# ==================== Dimension Validation ====================

def validate_dimensions_match(dim1: int, dim2: int, operation: str = "operation") -> None:
    """
    Validate that two dimensions match.
    
    Args:
        dim1: First dimension
        dim2: Second dimension
        operation: Name of operation for error message
    
    Raises:
        DimensionError: If dimensions don't match
    
    Example:
        >>> validate_dimensions_match(3, 3, "addition")
        >>> validate_dimensions_match(3, 4, "addition")  # Raises DimensionError
    """
    if dim1 != dim2:
        raise DimensionError(dim1, dim2, operation)


def validate_dimension_range(dim: int, min_dim: int, max_dim: int, 
                             name: str = "dimension") -> None:
    """
    Validate that a dimension is within allowed range.
    
    Args:
        dim: Dimension to validate
        min_dim: Minimum allowed dimension
        max_dim: Maximum allowed dimension
        name: Name for error messages
    
    Raises:
        DimensionError: If dimension is outside range
    
    Example:
        >>> validate_dimension_range(3, 2, 4, "vector")
        >>> validate_dimension_range(5, 2, 4, "vector")  # Raises error
    """
    if not (min_dim <= dim <= max_dim):
        raise DimensionError(
            dim, f"{min_dim}-{max_dim}", "dimension validation",
            f"{name} must be between {min_dim} and {max_dim}, got {dim}"
        )


# ==================== Matrix Validation ====================

def validate_matrix_dimensions(rows: int, cols: int) -> None:
    """
    Validate that matrix dimensions are positive integers.
    
    Args:
        rows: Number of rows
        cols: Number of columns
    
    Raises:
        ValidationError: If dimensions are invalid
    
    Example:
        >>> validate_matrix_dimensions(3, 4)
        >>> validate_matrix_dimensions(0, 4)  # Raises ValidationError
    """
    validate_integer(rows, "rows")
    validate_integer(cols, "cols")
    validate_positive(rows, "rows", strict=True)
    validate_positive(cols, "cols", strict=True)


def validate_square_matrix(rows: int, cols: int, operation: str = "operation") -> None:
    """
    Validate that a matrix is square.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        operation: Name of operation for error message
    
    Raises:
        NotSquareError: If matrix is not square
    
    Example:
        >>> validate_square_matrix(3, 3, "determinant")
        >>> validate_square_matrix(3, 4, "determinant")  # Raises NotSquareError
    """
    if rows != cols:
        raise NotSquareError((rows, cols), operation)


def validate_matrix_multiplication(m1_rows: int, m1_cols: int, 
                                   m2_rows: int, m2_cols: int) -> None:
    """
    Validate that two matrices can be multiplied.
    
    Args:
        m1_rows: Rows in first matrix
        m1_cols: Columns in first matrix
        m2_rows: Rows in second matrix
        m2_cols: Columns in second matrix
    
    Raises:
        ShapeError: If matrices cannot be multiplied
    
    Example:
        >>> validate_matrix_multiplication(2, 3, 3, 4)  # OK: (2x3) @ (3x4)
        >>> validate_matrix_multiplication(2, 3, 4, 2)  # Raises ShapeError
    """
    if m1_cols != m2_rows:
        raise ShapeError(
            (m1_rows, m1_cols),
            (m2_rows, m2_cols),
            "matrix multiplication",
            f"Cannot multiply ({m1_rows}x{m1_cols}) by ({m2_rows}x{m2_cols}): "
            f"inner dimensions must match"
        )


def validate_same_shape(shape1: Tuple[int, int], shape2: Tuple[int, int], 
                       operation: str = "operation") -> None:
    """
    Validate that two matrices have the same shape.
    
    Args:
        shape1: First matrix shape (rows, cols)
        shape2: Second matrix shape (rows, cols)
        operation: Name of operation for error message
    
    Raises:
        ShapeError: If shapes don't match
    
    Example:
        >>> validate_same_shape((3, 4), (3, 4), "addition")
        >>> validate_same_shape((3, 4), (4, 3), "addition")  # Raises ShapeError
    """
    if shape1 != shape2:
        raise ShapeError(shape1, shape2, operation)


# ==================== Vector Validation ====================

def validate_vector_dimension(dim: int, required: Union[int, Tuple[int, ...]], 
                              operation: str = "operation") -> None:
    """
    Validate that a vector has required dimension(s).
    
    Args:
        dim: Actual dimension
        required: Required dimension(s) - single int or tuple of allowed dimensions
        operation: Name of operation for error message
    
    Raises:
        DimensionError: If dimension doesn't match requirements
    
    Example:
        >>> validate_vector_dimension(3, 3, "cross product")
        >>> validate_vector_dimension(2, (2, 3), "operation")  # OK
        >>> validate_vector_dimension(4, 3, "cross product")  # Raises error
    """
    if isinstance(required, int):
        if dim != required:
            raise DimensionError(dim, required, operation)
    else:
        if dim not in required:
            raise DimensionError(
                dim, 
                f"one of {required}",
                operation,
                f"Vector dimension must be one of {required}, got {dim}"
            )


def validate_non_zero_magnitude(magnitude: float, tolerance: float = 1e-10) -> None:
    """
    Validate that a magnitude is not zero (within tolerance).
    
    Args:
        magnitude: Magnitude to check
        tolerance: Tolerance for zero check
    
    Raises:
        ZeroMagnitudeError: If magnitude is zero
    
    Example:
        >>> validate_non_zero_magnitude(1.5)
        >>> validate_non_zero_magnitude(0.0)  # Raises ZeroMagnitudeError
    """
    if abs(magnitude) < tolerance:
        raise ZeroMagnitudeError("Vector magnitude is zero")


# ==================== Data Validation ====================

def validate_list_not_empty(data: List, name: str = "data") -> None:
    """
    Validate that a list is not empty.
    
    Args:
        data: List to validate
        name: Name for error messages
    
    Raises:
        ValidationError: If list is empty
    """
    if not data:
        raise ValidationError(f"{name} cannot be empty")


def validate_rectangular_data(data: List[List], name: str = "data") -> Tuple[int, int]:
    """
    Validate that 2D list data is rectangular (all rows same length).
    
    Args:
        data: 2D list to validate
        name: Name for error messages
    
    Returns:
        Tuple of (rows, cols)
    
    Raises:
        ValidationError: If data is not rectangular
    
    Example:
        >>> validate_rectangular_data([[1, 2], [3, 4]])  # Returns (2, 2)
        >>> validate_rectangular_data([[1, 2], [3, 4, 5]])  # Raises error
    """
    validate_list_not_empty(data, name)
    
    rows = len(data)
    cols = len(data[0])
    
    for i, row in enumerate(data):
        if len(row) != cols:
            raise ValidationError(
                f"{name} must be rectangular: row 0 has {cols} elements, "
                f"but row {i} has {len(row)} elements"
            )
    
    return rows, cols


def validate_data_shape(data: List[List], expected_rows: int, expected_cols: int,
                       name: str = "data") -> None:
    """
    Validate that data matches expected shape.
    
    Args:
        data: 2D list to validate
        expected_rows: Expected number of rows
        expected_cols: Expected number of columns
        name: Name for error messages
    
    Raises:
        ValidationError: If shape doesn't match
    """
    actual_rows, actual_cols = validate_rectangular_data(data, name)
    
    if actual_rows != expected_rows or actual_cols != expected_cols:
        raise ValidationError(
            f"{name} shape mismatch: expected ({expected_rows}x{expected_cols}), "
            f"got ({actual_rows}x{actual_cols})"
        )


# ==================== Index Validation ====================

def validate_index(index: int, size: int, name: str = "index") -> None:
    """
    Validate that an index is within bounds.
    
    Args:
        index: Index to validate
        size: Size of the container
        name: Name for error messages
    
    Raises:
        ValidationError: If index is out of bounds
    
    Example:
        >>> validate_index(2, 5, "row")
        >>> validate_index(10, 5, "row")  # Raises ValidationError
    """
    if not (-size <= index < size):
        raise ValidationError(
            f"{name} {index} is out of bounds for size {size}"
        )


def validate_matrix_index(row: int, col: int, rows: int, cols: int) -> None:
    """
    Validate that matrix indices are within bounds.
    
    Args:
        row: Row index
        col: Column index
        rows: Number of rows in matrix
        cols: Number of columns in matrix
    
    Raises:
        ValidationError: If indices are out of bounds
    """
    validate_index(row, rows, "row")
    validate_index(col, cols, "column")


# ==================== Tolerance Validation ====================

def validate_tolerance(tolerance: float) -> None:
    """
    Validate that a tolerance value is positive.
    
    Args:
        tolerance: Tolerance value to validate
    
    Raises:
        ValidationError: If tolerance is not positive
    """
    validate_numeric(tolerance, "tolerance")
    validate_positive(tolerance, "tolerance", strict=True)


def approx_equal(a: float, b: float, tolerance: float = 1e-10) -> bool:
    """
    Check if two floats are approximately equal.
    
    Args:
        a: First value
        b: Second value
        tolerance: Tolerance for comparison
    
    Returns:
        True if |a - b| < tolerance
    
    Example:
        >>> approx_equal(1.0, 1.0000000001)
        True
        >>> approx_equal(1.0, 1.1)
        False
    """
    return abs(a - b) < tolerance


def approx_zero(value: float, tolerance: float = 1e-10) -> bool:
    """
    Check if a value is approximately zero.
    
    Args:
        value: Value to check
        tolerance: Tolerance for comparison
    
    Returns:
        True if |value| < tolerance
    
    Example:
        >>> approx_zero(0.0000000001)
        True
        >>> approx_zero(0.1)
        False
    """
    return abs(value) < tolerance
