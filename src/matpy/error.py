"""
Custom exceptions for the matpy library.

This module provides specialized exception classes to make error handling
more precise and code cleaner throughout the matpy package.
"""


class MatPyError(Exception):
    """Base exception for all matpy errors."""
    pass


class ValidationError(MatPyError):
    """Raised when input data is invalid or fails validation."""
    pass


class DimensionError(MatPyError):
    """Raised when vector/matrix dimensions do not match for an operation."""
    
    def __init__(self, expected=None, actual=None, message=None):
        if message is None and expected and actual:
            message = f"Dimension mismatch: expected {expected}, got {actual}"
        elif message is None:
            message = "Incompatible dimensions for this operation"
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class ShapeError(MatPyError):
    """Raised when matrix shapes are incompatible for an operation."""
    
    def __init__(self, shape1=None, shape2=None, operation=None, message=None):
        if message is None and shape1 and shape2 and operation:
            message = f"Incompatible shapes for {operation}: {shape1} and {shape2}"
        elif message is None:
            message = "Matrix shapes are incompatible for this operation"
        super().__init__(message)
        self.shape1 = shape1
        self.shape2 = shape2
        self.operation = operation


class NotSquareError(MatPyError):
    """Raised when an operation requires a square matrix but matrix is not square."""
    
    def __init__(self, rows=None, cols=None, message=None):
        if message is None and rows and cols:
            message = f"Operation requires square matrix, got {rows}x{cols}"
        elif message is None:
            message = "This operation requires a square matrix"
        super().__init__(message)
        self.rows = rows
        self.cols = cols


class SingularMatrixError(MatPyError):
    """Raised when a matrix is singular (non-invertible, determinant = 0)."""
    
    def __init__(self, message=None):
        if message is None:
            message = "Matrix is singular and cannot be inverted (determinant = 0)"
        super().__init__(message)


class IndexError(MatPyError):
    """Raised when an index is out of valid range."""
    
    def __init__(self, index=None, valid_range=None, message=None):
        if message is None and index is not None and valid_range:
            message = f"Index {index} out of range {valid_range}"
        elif message is None:
            message = "Index out of valid range"
        super().__init__(message)
        self.index = index
        self.valid_range = valid_range


class ZeroMagnitudeError(MatPyError):
    """Raised when an operation requires non-zero magnitude but vector has zero magnitude."""
    
    def __init__(self, message=None):
        if message is None:
            message = "Cannot perform operation on zero-magnitude vector"
        super().__init__(message)


class NotInvertibleError(MatPyError):
    """Raised when attempting to invert a non-invertible matrix."""
    
    def __init__(self, reason=None, message=None):
        if message is None and reason:
            message = f"Matrix is not invertible: {reason}"
        elif message is None:
            message = "Matrix is not invertible"
        super().__init__(message)
        self.reason = reason


class ComputationError(MatPyError):
    """Raised when a numerical computation fails or produces invalid results."""
    
    def __init__(self, operation=None, message=None):
        if message is None and operation:
            message = f"Computation failed during {operation}"
        elif message is None:
            message = "Numerical computation failed"
        super().__init__(message)
        self.operation = operation


class NotImplementedError(MatPyError):
    """Raised when a feature is not yet implemented."""
    
    def __init__(self, feature=None, message=None):
        if message is None and feature:
            message = f"{feature} is not yet implemented"
        elif message is None:
            message = "This feature is not yet implemented"
        super().__init__(message)
        self.feature = feature


# Convenience functions for common error checks
def require_square(matrix, operation="This operation"):
    """
    Check if matrix is square, raise NotSquareError if not.
    
    Args:
        matrix: Matrix object to check
        operation: Description of the operation requiring square matrix
    
    Raises:
        NotSquareError: If matrix is not square
    """
    if matrix.rows != matrix.cols:
        raise NotSquareError(
            matrix.rows, 
            matrix.cols, 
            f"{operation} requires a square matrix"
        )


def require_same_dimensions(obj1, obj2, operation="This operation"):
    """
    Check if two objects have same dimensions.
    
    Args:
        obj1: First vector/matrix
        obj2: Second vector/matrix
        operation: Description of the operation
    
    Raises:
        DimensionError: If dimensions don't match
    """
    if hasattr(obj1, 'rows') and hasattr(obj2, 'rows'):
        # Matrices
        if obj1.rows != obj2.rows or obj1.cols != obj2.cols:
            raise ShapeError(
                (obj1.rows, obj1.cols),
                (obj2.rows, obj2.cols),
                operation
            )
    else:
        # Vectors or other objects
        if (obj1.x, obj1.y, obj1.z) != (obj2.x, obj2.y, obj2.z):
            raise DimensionError(
                message=f"{operation} requires objects with matching dimensions"
            )


def require_compatible_for_multiplication(matrix1, matrix2):
    """
    Check if two matrices can be multiplied (m1 @ m2).
    
    Args:
        matrix1: First matrix
        matrix2: Second matrix
    
    Raises:
        ShapeError: If matrices cannot be multiplied
    """
    if matrix1.cols != matrix2.rows:
        raise ShapeError(
            (matrix1.rows, matrix1.cols),
            (matrix2.rows, matrix2.cols),
            "matrix multiplication",
            f"Cannot multiply {matrix1.rows}x{matrix1.cols} by {matrix2.rows}x{matrix2.cols} matrix"
        )


def require_non_zero(value, name="Value"):
    """
    Check if value is non-zero.
    
    Args:
        value: Value to check
        name: Name of the value for error message
    
    Raises:
        ZeroDivisionError: If value is zero or near-zero
    """
    if abs(value) < 1e-10:
        raise ZeroDivisionError(f"{name} cannot be zero")


def require_non_singular(matrix):
    """
    Check if matrix is non-singular (invertible).
    
    Args:
        matrix: Matrix to check
    
    Raises:
        SingularMatrixError: If matrix is singular
    """
    det = matrix.determinant()
    if abs(det) < 1e-10:
        raise SingularMatrixError()


__all__ = [
    'MatPyError',
    'ValidationError',
    'DimensionError',
    'ShapeError',
    'NotSquareError',
    'SingularMatrixError',
    'IndexError',
    'ZeroMagnitudeError',
    'NotInvertibleError',
    'ComputationError',
    'NotImplementedError',
    'require_square',
    'require_same_dimensions',
    'require_compatible_for_multiplication',
    'require_non_zero',
    'require_non_singular',
]

