"""
Linear systems and differential equations solver module.

This module provides functions for solving:
- Linear systems of equations (Ax = b)
- Systems of differential equations (both homogeneous and non-homogeneous)
"""

from __future__ import annotations
from typing import List, Tuple, Union
import math

from .core import Matrix
from .ops import identity, zeros
from ..vector.core import Vector
from ..error import SingularMatrixError, NotImplementedError
from ..core import validate
from ..core.math_utils import euler_method, runge_kutta_4

# Default tolerance for floating point comparisons
DEFAULT_TOLERANCE = 1e-10


# ==================== Linear Systems ====================

def gaussian(A: Matrix, b: Union[Vector, List[float]]) -> Vector:
    """
    Solve the linear system Ax = b using Gaussian elimination with partial pivoting.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (Vector or list of length n)
    
    Returns:
        Solution vector x as Vector object
    
    Raises:
        NotSquareError: If A is not square
        ShapeError: If b length doesn't match A rows
        SingularMatrixError: If system has no unique solution
    
    Example:
        >>> # Solve: 2x + y = 5, x + 3y = 7
        >>> A = Matrix(2, 2, [[2, 1], [1, 3]])
        >>> b = Vector(5, 7)  # or [5, 7]
        >>> x = gaussian(A, b)
        >>> # x ≈ Vector(1.0, 3.0)
    """
    # Convert Vector to list if needed
    b_list = list(b.components) if isinstance(b, Vector) else b
    
    validate.validate_square_matrix(A.rows, A.cols, "solve linear system")
    validate.validate_dimensions_match(len(b_list), A.rows, "system solve (b vector)")
    
    n = A.rows
    # Create augmented matrix [A|b]
    augmented_data = [A.data[i][:] + [b_list[i]] for i in range(n)]
    
    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot (largest absolute value in column)
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented_data[row][col]) > abs(augmented_data[max_row][col]):
                max_row = row
        
        # Check for singular matrix
        if validate.approx_zero(augmented_data[max_row][col]):
            raise SingularMatrixError(
                "System has no unique solution (matrix is singular or inconsistent)"
            )
        
        # Swap rows
        if max_row != col:
            augmented_data[col], augmented_data[max_row] = augmented_data[max_row], augmented_data[col]
        
        # Eliminate below
        pivot = augmented_data[col][col]
        for row in range(col + 1, n):
            factor = augmented_data[row][col] / pivot
            for j in range(col, n + 1):
                augmented_data[row][j] -= factor * augmented_data[col][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented_data[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented_data[i][j] * x[j]
        x[i] /= augmented_data[i][i]
    
    return Vector(*x)


def lu(A: Matrix, b: Union[Vector, List[float]]) -> Vector:
    """
    Solve linear system using LU decomposition.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (Vector or list of length n)
    
    Returns:
        Solution vector x as Vector object
    
    Note:
        This uses LU decomposition: A = LU, then solves Ly = b, Ux = y
    """
    # Convert Vector to list if needed
    b_list = list(b.components) if isinstance(b, Vector) else b
    
    L, U = lu_decomposition(A)
    
    # Forward substitution: Ly = b
    n = A.rows
    y = [0.0] * n
    for i in range(n):
        y[i] = b_list[i]
        for j in range(i):
            y[i] -= L.data[i][j] * y[j]
        y[i] /= L.data[i][i]
    
    # Back substitution: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U.data[i][j] * x[j]
        x[i] /= U.data[i][i]
    
    return Vector(*x)


def lu_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Perform LU decomposition: A = LU where L is lower triangular, U is upper triangular.
    
    Args:
        A: Square matrix
    
    Returns:
        Tuple (L, U) of lower and upper triangular matrices
    
    Raises:
        NotSquareError: If A is not square
        SingularMatrixError: If decomposition fails
    """
    validate.validate_square_matrix(A.rows, A.cols, "LU decomposition")
    
    n = A.rows
    L = zeros(n, n)
    U = zeros(n, n)
    
    for i in range(n):
        # Upper triangular
        for k in range(i, n):
            sum_val = sum(L.data[i][j] * U.data[j][k] for j in range(i))
            U.data[i][k] = A.data[i][k] - sum_val
        
        # Lower triangular
        for k in range(i, n):
            if i == k:
                L.data[i][i] = 1.0
            else:
                if validate.approx_zero(U.data[i][i]):
                    raise SingularMatrixError("LU decomposition failed (singular matrix)")
                sum_val = sum(L.data[k][j] * U.data[j][i] for j in range(i))
                L.data[k][i] = (A.data[k][i] - sum_val) / U.data[i][i]
    
    return L, U


def cramers(A: Matrix, b: Union[Vector, List[float]]) -> Vector:
    """
    Solve linear system using Cramer's rule.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (Vector or list of length n)
    
    Returns:
        Solution vector x as Vector object
    
    Note:
        Cramer's rule: x_i = det(A_i) / det(A)
        where A_i is A with column i replaced by b.
        This is inefficient for large systems but exact for small ones.
    
    Raises:
        SingularMatrixError: If det(A) = 0
    """
    # Convert Vector to list if needed
    b_list = list(b.components) if isinstance(b, Vector) else b
    
    det_A = A.determinant()
    
    if validate.approx_zero(det_A):
        raise SingularMatrixError("Cannot solve system (determinant is zero)")
    
    n = A.rows
    x = [0.0] * n
    
    for i in range(n):
        # Create matrix with column i replaced by b
        A_i_data = [row[:] for row in A.data]
        for j in range(n):
            A_i_data[j][i] = b_list[j]
        A_i = Matrix(n, n, A_i_data)
        
        x[i] = A_i.determinant() / det_A
    
    return Vector(*x)


def solve_least_squares(A: Matrix, b: Union[Vector, List[float]]) -> Vector:
    """
    Solve overdetermined system using least squares: min ||Ax - b||².
    
    Args:
        A: Coefficient matrix (m x n, typically m > n)
        b: Right-hand side vector (Vector or list of length m)
    
    Returns:
        Least squares solution vector x as Vector object (length n)
    
    Note:
        Solves the normal equations: (A^T A)x = A^T b
    
    Example:
        >>> # Fit line y = mx + c to points
        >>> # Points: (0,1), (1,2), (2,4), (3,5)
        >>> A = Matrix(4, 2, [[0, 1], [1, 1], [2, 1], [3, 1]])
        >>> b = Vector(1, 2, 4, 5)  # or [1, 2, 4, 5]
        >>> x = solve_least_squares(A, b)  # Vector(m, c)
    """
    # Convert Vector to list if needed
    b_list = list(b.components) if isinstance(b, Vector) else b
    
    validate.validate_dimensions_match(len(b_list), A.rows, "least squares solve (b vector)")
    
    # Compute A^T A and A^T b
    A_T = A.transpose()
    A_T_A = A_T @ A
    
    # A^T b
    A_T_b = [sum(A_T.data[i][j] * b_list[j] for j in range(len(b_list))) for i in range(A.cols)]
    
    # Solve (A^T A)x = A^T b
    return gaussian(A_T_A, A_T_b)


# ==================== Systems of Differential Equations ====================

def solve_linear_ode_system_homogeneous(
    A: Matrix, 
    x0: List[float], 
    t: float,
    method: str = "rk4",
    dt: float = 0.01
) -> List[float]:
    """
    Solve homogeneous linear ODE system: dx/dt = Ax with initial condition x(0) = x₀.
    
    Args:
        A: Coefficient matrix (n x n, constant)
        x0: Initial values x₀ (length n)
        t: Time value at which to evaluate solution
        method: Solution method - 'rk4', 'euler', or 'analytical' (default: 'rk4')
        dt: Time step for numerical methods (default: 0.01)
    
    Returns:
        Solution vector x(t)
    
    Note:
        - 'rk4': Uses 4th-order Runge-Kutta (most accurate)
        - 'euler': Uses Euler's method (faster but less accurate)
        - 'analytical': Uses matrix exponential (exact for linear systems)
    
    Example:
        >>> # Solve: dx/dt = -2x + y, dy/dt = x - 2y
        >>> A = Matrix(2, 2, [[-2, 1], [1, -2]])
        >>> x0 = [1, 0]
        >>> x_t = solve_linear_ode_system_homogeneous(A, x0, 1.0, method='rk4')
    """
    validate.validate_square_matrix(A.rows, A.cols, "ODE system solve")
    validate.validate_dimensions_match(len(x0), A.rows, "ODE initial conditions")
    
    # Define derivative function: dx/dt = Ax
    def f(t_val, x):
        result = [0.0] * A.rows
        for i in range(A.rows):
            for j in range(A.cols):
                result[i] += A.data[i][j] * x[j]
        return result
    
    if method.lower() == "rk4":
        # Use Runge-Kutta 4 from math_utils
        # Returns (times, solutions) - we only need the final solution
        times, solutions = runge_kutta_4(f, x0, 0, t, dt)
        return solutions[-1]
    
    elif method.lower() == "euler":
        # Use Euler method from math_utils
        # Returns (times, solutions) - we only need the final solution
        times, solutions = euler_method(f, x0, 0, t, dt)
        return solutions[-1]
    
    elif method.lower() == "analytical":
        # Use analytical solution via eigenvalue method
        # Get eigenvalues from the Matrix class
        eigenvals = Matrix.eigenvalues(A)
        
        if A.rows == 1:
            # Simple exponential: x(t) = x₀ * exp(λ*t)
            λ = eigenvals[0]
            # Handle both real and complex eigenvalues
            if isinstance(λ, complex):
                # Use cmath for complex exponential
                import cmath
                result = x0[0] * cmath.exp(λ * t)
                # Return real part for real-valued ODEs
                return [result.real]
            else:
                return [x0[0] * math.exp(λ * t)]
        
        elif A.rows == 2:
            # Use eigenvalue method for 2x2 systems
            return _solve_2x2_ode_eigenvalue(A, x0, t)
        
        # General case: use matrix exponential
        exp_At = matrix_exp(A, t)
        
        # Multiply exp(At) by initial conditions
        result = [0.0] * A.rows
        for i in range(A.rows):
            result[i] = sum(exp_At.data[i][j] * x0[j] for j in range(A.cols))
        
        return result
    
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'rk4', 'euler', or 'analytical'.")


def _solve_2x2_ode_eigenvalue(A: Matrix, x0: List[float], t: float) -> List[float]:
    """
    Solve 2x2 ODE system using eigenvalue method from Matrix class.
    
    Uses Matrix.eigenvalues() and Matrix.eigenvectors() static methods
    to calculate eigenvalues and eigenvectors, then applies the analytical
    solution formula. Falls back to matrix exponential for complex or repeated eigenvalues.
    """
    # Get eigenvalues and eigenvectors from the Matrix class static methods
    eigenvals = Matrix.eigenvalues(A)
    eigenvecs = Matrix.eigenvectors(A)
    
    λ1, λ2 = eigenvals[0], eigenvals[1]
    v1, v2 = eigenvecs[0], eigenvecs[1]
    
    # Check discriminant to see if eigenvalues are real and distinct
    disc = A.trace() ** 2 - 4 * A.determinant()
    
    # Only use eigenvalue/eigenvector method for real, distinct eigenvalues
    if disc > DEFAULT_TOLERANCE and abs(λ1 - λ2) > DEFAULT_TOLERANCE:
        # Solve for coefficients: x₀ = c₁*v₁ + c₂*v₂
        # Construct matrix V with eigenvectors as columns: V = [v₁ | v₂]
        V = Matrix(2, 2, [[v1[0], v2[0]], [v1[1], v2[1]]])
        
        try:
            # Use Cramer's rule to solve V*c = x₀ for coefficients c = [c₁, c₂]
            coeffs = cramers(V, x0)
            c1, c2 = coeffs.components[0], coeffs.components[1]
            
            # Solution: x(t) = c₁*exp(λ₁*t)*v₁ + c₂*exp(λ₂*t)*v₂
            exp_λ1t = math.exp(λ1 * t)
            exp_λ2t = math.exp(λ2 * t)
            
            x = c1 * exp_λ1t * v1[0] + c2 * exp_λ2t * v2[0]
            y = c1 * exp_λ1t * v1[1] + c2 * exp_λ2t * v2[1]
            
            return [x, y]
        except SingularMatrixError:
            # Eigenvectors are linearly dependent, fall through to matrix exponential
            pass
    
    # Complex or repeated eigenvalues - use matrix exponential
    exp_At = matrix_exp(A, t)
    result = [0.0] * 2
    for i in range(2):
        result[i] = sum(exp_At.data[i][j] * x0[j] for j in range(2))
    return result

def _solve_3x3_ode_eigenvalue(A: Matrix, x0: List[float], t: float) -> List[float]:
    """
    Placeholder for 3x3 ODE solver using eigenvalue method.
    
    This is more complex due to potential for repeated eigenvalues and complex eigenvalues.
    For now, this function is not implemented and will raise NotImplementedError.
    """
    raise NotImplementedError("3x3 ODE solver using eigenvalue method is not implemented yet.")

def matrix_exp(A: Matrix, t: float = 1.0, terms: int = 20) -> Matrix:
    """
    Calculate matrix exponential exp(At) using Taylor series.
    
    Args:
        A: Square matrix
        t: Scalar multiplier (default 1.0)
        terms: Number of terms in Taylor series (default 20)
    
    Returns:
        exp(At) ≈ I + At + (At)²/2! + (At)³/3! + ...
    
    Note:
        More terms give better accuracy but take longer to compute.
    """
    if not A.is_square():
        from ..error import NotSquareError
        raise NotSquareError((A.rows, A.cols), "matrix exponential")
    
    n = A.rows
    result = identity(n)
    
    # Scale A by t
    At = A * t
    
    # Series expansion
    term = identity(n)
    factorial = 1.0
    
    for k in range(1, terms):
        factorial *= k
        term = term @ At
        result = result + term * (1.0 / factorial)
    
    return result

def solve_linear(A, b, method="gaussian"):
    """
    General linear system solver with method selection.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (Vector or list of length n)
        method: Solution method - 'gaussian', 'lu', or 'cramer' (default: 'gaussian')
    
    Returns:
        Solution vector x as Vector object
    
    Example:
        >>> A = Matrix(2, 2, [[2, 1], [1, 3]])
        >>> b = [5, 7]
        >>> x = solve_linear(A, b, method='gaussian')
    """
    if method == "gaussian":
        return gaussian(A, b)
    elif method == "lu":
        return lu(A, b)
    elif method == "cramer":
        return cramers(A, b)
    else:
        raise ValueError(f"Unknown method '{method}' for solving linear system")