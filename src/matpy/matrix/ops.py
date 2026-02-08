"""
Matrix operations module.

This module provides standalone functions for common matrix operations.
These functions complement the Matrix class methods in core.py.
"""

from __future__ import annotations
from typing import List, Tuple

from .core import Matrix
from ..core import validate, utils

# Default tolerance for floating point comparisons
DEFAULT_TOLERANCE = 1e-10


# ==================== Matrix Creation ====================

def zeros(rows: int, cols: int) -> Matrix:
    """
    Create a matrix filled with zeros.
    
    Args:
        rows: Number of rows
        cols: Number of columns
    
    Returns:
        Matrix filled with zeros
    """
    return Matrix(rows, cols)


def ones(rows: int, cols: int) -> Matrix:
    """
    Create a matrix filled with ones.
    
    Args:
        rows: Number of rows
        cols: Number of columns
    
    Returns:
        Matrix filled with ones
    """
    data = [[1 for _ in range(cols)] for _ in range(rows)]
    return Matrix(rows, cols, data)


def identity(size: int) -> Matrix:
    """
    Create an identity matrix.
    
    Args:
        size: Size of the square identity matrix
    
    Returns:
        Identity matrix with 1s on diagonal, 0s elsewhere
    """
    data = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    return Matrix(size, size, data)


def diagonal(values: List[float]) -> Matrix:
    """
    Create a diagonal matrix from a list of values.
    
    Args:
        values: List of values for the diagonal
    
    Returns:
        Diagonal matrix with specified values on diagonal, 0s elsewhere
    """
    size = len(values)
    data = [[values[i] if i == j else 0 for j in range(size)] for i in range(size)]
    return Matrix(size, size, data)


def from_rows(rows: List[List[float]]) -> Matrix:
    """
    Create a matrix from a list of rows.
    
    Args:
        rows: List of rows, where each row is a list of values
    
    Returns:
        Matrix created from the rows
    
    Raises:
        ValueError: If rows have inconsistent lengths
    """
    num_rows, num_cols = validate.validate_rectangular_data(rows, "rows")
    return Matrix(num_rows, num_cols, rows)


def from_columns(cols: List[List[float]]) -> Matrix:
    """
    Create a matrix from a list of columns.
    
    Args:
        cols: List of columns, where each column is a list of values
    
    Returns:
        Matrix created from the columns
    
    Raises:
        ValueError: If columns have inconsistent lengths
    """
    num_cols, num_rows = validate.validate_rectangular_data(cols, "columns")
    
    data = [[cols[j][i] for j in range(num_cols)] for i in range(num_rows)]
    return Matrix(num_rows, num_cols, data)

# ==================== Advanced Operations ====================

def power(matrix: Matrix, exponent: int) -> Matrix:
    """
    Raise matrix to a power using repeated multiplication.
    
    Args:
        matrix: The matrix (must be square)
        exponent: The exponent (non-negative integer)
    
    Returns:
        Matrix raised to the power
    """
    return matrix ** exponent

def hadamard_product(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Calculate the Hadamard product (element-wise multiplication).
    
    Args:
        m1: First matrix
        m2: Second matrix (must have same dimensions as m1)
    
    Returns:
        Element-wise product of the matrices
    
    Raises:
        ShapeError: If matrices have different dimensions
    """
    from ..error import ShapeError
    
    if m1.rows != m2.rows or m1.cols != m2.cols:
        raise ShapeError(
            (m1.rows, m1.cols),
            (m2.rows, m2.cols),
            "Hadamard product"
        )
    
    result_data = [
        [m1.data[i][j] * m2.data[i][j] for j in range(m1.cols)]
        for i in range(m1.rows)
    ]
    return Matrix(m1.rows, m1.cols, result_data)

def kronecker_product(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Calculate the Kronecker product of two matrices.
    
    Args:
        m1: First matrix
        m2: Second matrix
    
    Returns:
        Kronecker product (block matrix)
    """
    result_rows = m1.rows * m2.rows
    result_cols = m1.cols * m2.cols
    
    result_data = [[0 for _ in range(result_cols)] for _ in range(result_rows)]
    
    for i in range(m1.rows):
        for j in range(m1.cols):
            for k in range(m2.rows):
                for l in range(m2.cols):
                    result_data[i * m2.rows + k][j * m2.cols + l] = m1.data[i][j] * m2.data[k][l]
    
    return Matrix(result_rows, result_cols, result_data)

def frobenius_norm(matrix: Matrix) -> float:
    """
    Calculate the Frobenius norm of a matrix.
    
    Args:
        matrix: The matrix
    
    Returns:
        The Frobenius norm (square root of sum of squares of all elements)
    """
    return abs(matrix)

def REF(matrix: Matrix) -> Matrix:
    """
    Convert matrix to row echelon form using Gaussian elimination.
    
    Args:
        matrix: The matrix
    
    Returns:
        Matrix in row echelon form
    """
    result = matrix.__copy__()
    
    pivot_row = 0
    for col in range(result.cols):
        # Find pivot
        found_pivot = False
        for row in range(pivot_row, result.rows):
            if abs(result.data[row][col]) > DEFAULT_TOLERANCE:
                # Swap rows
                if row != pivot_row:
                    result.data[pivot_row], result.data[row] = result.data[row], result.data[pivot_row]
                found_pivot = True
                break
        
        if not found_pivot:
            continue
        
        # Scale pivot row
        pivot = result.data[pivot_row][col]
        for j in range(result.cols):
            result.data[pivot_row][j] /= pivot
        
        # Eliminate below
        for row in range(pivot_row + 1, result.rows):
            factor = result.data[row][col]
            for j in range(result.cols):
                result.data[row][j] -= factor * result.data[pivot_row][j]
        
        pivot_row += 1
    
    return result

def RREF(matrix: Matrix) -> Matrix:
    """
    Convert matrix to reduced row echelon form.
    
    Args:
        matrix: The matrix
    
    Returns:
        Matrix in reduced row echelon form
    """
    result = REF(matrix)
    
    # Back substitution to make it reduced
    for i in range(min(result.rows, result.cols) - 1, -1, -1):
        # Find leading 1
        leading_col = None
        for j in range(result.cols):
            if abs(result.data[i][j] - 1.0) < DEFAULT_TOLERANCE:
                leading_col = j
                break
        
        if leading_col is None:
            continue
        
        # Eliminate above
        for row in range(i):
            factor = result.data[row][leading_col]
            for j in range(result.cols):
                result.data[row][j] -= factor * result.data[i][j]
    
    return result

def concatenate_horizontal(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Concatenate two matrices horizontally (side by side).
    
    Args:
        m1: First matrix
        m2: Second matrix (must have same number of rows)
    
    Returns:
        Concatenated matrix
    
    Raises:
        ShapeError: If matrices have different number of rows
    """
    from ..error import ShapeError
    
    if m1.rows != m2.rows:
        raise ShapeError(
            (m1.rows, m1.cols),
            (m2.rows, m2.cols),
            "horizontal concatenation",
            f"Matrices must have same number of rows for horizontal concatenation"
        )
    
    result_data = [
        m1.data[i] + m2.data[i]
        for i in range(m1.rows)
    ]
    return Matrix(m1.rows, m1.cols + m2.cols, result_data)

def concatenate_vertical(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Concatenate two matrices vertically (stacked).
    
    Args:
        m1: First matrix
        m2: Second matrix (must have same number of columns)
    
    Returns:
        Concatenated matrix
    
    Raises:
        ShapeError: If matrices have different number of columns
    """
    from ..error import ShapeError
    
    if m1.cols != m2.cols:
        raise ShapeError(
            (m1.rows, m1.cols),
            (m2.rows, m2.cols),
            "vertical concatenation",
            f"Matrices must have same number of columns for vertical concatenation"
        )
    
    result_data = m1.data + m2.data
    return Matrix(m1.rows + m2.rows, m1.cols, result_data)

