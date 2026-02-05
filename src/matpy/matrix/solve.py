"""
Linear systems and differential equations solver module.

This module provides functions for solving:
- Linear systems of equations (Ax = b)
- Systems of differential equations (both homogeneous and non-homogeneous)
"""

from __future__ import annotations
from typing import List, Tuple, Callable, Optional
import math

from .core import Matrix
from .ops import identity, zeros, from_rows
from ..error import ShapeError, SingularMatrixError, ValidationError, MatPyNotImplementedError
from ..core import validate, utils


# ==================== Linear Systems ====================

def solve_linear_system(A: Matrix, b: List[float]) -> List[float]:
    """
    Solve the linear system Ax = b using Gaussian elimination with partial pivoting.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (length n)
    
    Returns:
        Solution vector x
    
    Raises:
        NotSquareError: If A is not square
        ShapeError: If b length doesn't match A rows
        SingularMatrixError: If system has no unique solution
    
    Example:
        >>> # Solve: 2x + y = 5, x + 3y = 7
        >>> A = Matrix(2, 2, [[2, 1], [1, 3]])
        >>> b = [5, 7]
        >>> x = solve_linear_system(A, b)
        >>> # x ≈ [1.0, 3.0]
    """
    validate.validate_square_matrix(A.rows, A.cols, "solve linear system")
    validate.validate_dimensions_match(len(b), A.rows, "system solve (b vector)")
    
    n = A.rows
    # Create augmented matrix [A|b]
    augmented_data = [A.data[i][:] + [b[i]] for i in range(n)]
    
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
    
    return x


def solve_lu(A: Matrix, b: List[float]) -> List[float]:
    """
    Solve linear system using LU decomposition.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (length n)
    
    Returns:
        Solution vector x
    
    Note:
        This uses LU decomposition: A = LU, then solves Ly = b, Ux = y
    """
    L, U = lu_decomposition(A)
    
    # Forward substitution: Ly = b
    n = A.rows
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
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
    
    return x


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


def solve_cramer(A: Matrix, b: List[float]) -> List[float]:
    """
    Solve linear system using Cramer's rule.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (length n)
    
    Returns:
        Solution vector x
    
    Note:
        Cramer's rule: x_i = det(A_i) / det(A)
        where A_i is A with column i replaced by b.
        This is inefficient for large systems but exact for small ones.
    
    Raises:
        SingularMatrixError: If det(A) = 0
    """
    det_A = A.determinant()
    
    if validate.approx_zero(det_A):
        raise SingularMatrixError("Cannot solve system (determinant is zero)")
    
    n = A.rows
    x = [0.0] * n
    
    for i in range(n):
        # Create matrix with column i replaced by b
        A_i_data = [row[:] for row in A.data]
        for j in range(n):
            A_i_data[j][i] = b[j]
        A_i = Matrix(n, n, A_i_data)
        
        x[i] = A_i.determinant() / det_A
    
    return x


def solve_least_squares(A: Matrix, b: List[float]) -> List[float]:
    """
    Solve overdetermined system using least squares: min ||Ax - b||².
    
    Args:
        A: Coefficient matrix (m x n, typically m > n)
        b: Right-hand side vector (length m)
    
    Returns:
        Least squares solution vector x (length n)
    
    Note:
        Solves the normal equations: (A^T A)x = A^T b
    
    Example:
        >>> # Fit line y = mx + c to points
        >>> # Points: (0,1), (1,2), (2,4), (3,5)
        >>> A = Matrix(4, 2, [[0, 1], [1, 1], [2, 1], [3, 1]])
        >>> b = [1, 2, 4, 5]
        >>> x = solve_least_squares(A, b)  # [m, c]
    """
    validate.validate_dimensions_match(len(b), A.rows, "least squares solve (b vector)")
    
    # Compute A^T A and A^T b
    A_T = A.transpose()
    A_T_A = A_T @ A
    
    # A^T b
    A_T_b = [sum(A_T.data[i][j] * b[j] for j in range(len(b))) for i in range(A.cols)]
    
    # Solve (A^T A)x = A^T b
    return solve_linear_system(A_T_A, A_T_b)


# ==================== Systems of Differential Equations ====================

def solve_linear_ode_system_homogeneous(A: Matrix, initial_conditions: List[float], t: float) -> List[float]:
    """
    Solve homogeneous linear ODE system: dx/dt = Ax with initial condition x(0) = x₀.
    
    Args:
        A: Coefficient matrix (n x n, constant)
        initial_conditions: Initial values x₀ (length n)
        t: Time value at which to evaluate solution
    
    Returns:
        Solution vector x(t)
    
    Note:
        Solution is x(t) = exp(At) x₀
        For 2x2 matrices, uses eigenvalue method when possible.
        For general case, uses matrix exponential via series expansion.
    
    Example:
        >>> # Solve: dx/dt = -2x + y, dy/dt = x - 2y
        >>> A = Matrix(2, 2, [[-2, 1], [1, -2]])
        >>> x0 = [1, 0]
        >>> x_t = solve_linear_ode_system_homogeneous(A, x0, 1.0)
    """
    validate.validate_square_matrix(A.rows, A.cols, "ODE system solve")
    validate.validate_dimensions_match(len(initial_conditions), A.rows, "ODE initial conditions")
    
    # For small matrices (1x1, 2x2), try analytical solution
    if A.rows == 1:
        # Simple exponential: x(t) = x₀ * exp(a*t)
        a = A.data[0][0]
        return [initial_conditions[0] * math.exp(a * t)]
    
    elif A.rows == 2:
        # Try eigenvalue method for 2x2
        try:
            return _solve_2x2_ode_eigenvalue(A, initial_conditions, t)
        except:
            # Fall back to matrix exponential
            pass
    
    # General case: use matrix exponential
    exp_At = matrix_exponential(A, t)
    
    # Multiply exp(At) by initial conditions
    result = [0.0] * A.rows
    for i in range(A.rows):
        result[i] = sum(exp_At.data[i][j] * initial_conditions[j] for j in range(A.cols))
    
    return result


def _solve_2x2_ode_eigenvalue(A: Matrix, x0: List[float], t: float) -> List[float]:
    """
    Solve 2x2 ODE system using eigenvalue method.
    
    For A = [[a, b], [c, d]], eigenvalues are:
    λ = (a+d)/2 ± sqrt((a+d)²/4 - (ad-bc))
    """
    a, b = A.data[0][0], A.data[0][1]
    c, d = A.data[1][0], A.data[1][1]
    
    trace = a + d
    det = a * d - b * c
    discriminant = trace * trace / 4 - det
    
    if discriminant >= 0:
        # Real eigenvalues
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = trace / 2 + sqrt_disc
        lambda2 = trace / 2 - sqrt_disc
        
        # For simplicity, use matrix exponential formula
        # This is a simplified version
        if abs(lambda1 - lambda2) > 1e-10:
            # Distinct eigenvalues
            exp_l1t = math.exp(lambda1 * t)
            exp_l2t = math.exp(lambda2 * t)
            
            # Use formula: exp(At) = (exp(λ₁t)(A-λ₂I) - exp(λ₂t)(A-λ₁I))/(λ₁-λ₂)
            c1 = (exp_l1t - exp_l2t) / (lambda1 - lambda2)
            c2 = (lambda1 * exp_l2t - lambda2 * exp_l1t) / (lambda1 - lambda2)
            
            exp_At_00 = c2 + c1 * a
            exp_At_01 = c1 * b
            exp_At_10 = c1 * c
            exp_At_11 = c2 + c1 * d
            
            x = exp_At_00 * x0[0] + exp_At_01 * x0[1]
            y = exp_At_10 * x0[0] + exp_At_11 * x0[1]
            
            return [x, y]
    
    # Fall back to series expansion
    raise MatPyNotImplementedError("Complex eigenvalues require matrix exponential")


def matrix_exponential(A: Matrix, t: float = 1.0, terms: int = 20) -> Matrix:
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


def solve_linear_ode_system_nonhomogeneous(
    A: Matrix,
    f: Callable[[float], List[float]],
    initial_conditions: List[float],
    t: float,
    dt: float = 0.01
) -> List[float]:
    """
    Solve non-homogeneous linear ODE system: dx/dt = Ax + f(t) with x(0) = x₀.
    
    Args:
        A: Coefficient matrix (n x n)
        f: Forcing function f(t) returning vector of length n
        initial_conditions: Initial values x₀ (length n)
        t: Time value at which to evaluate solution
        dt: Time step for numerical integration (default 0.01)
    
    Returns:
        Solution vector x(t)
    
    Note:
        Uses 4th-order Runge-Kutta method for numerical integration.
    
    Example:
        >>> # Solve: dx/dt = -x + sin(t), x(0) = 1
        >>> A = Matrix(1, 1, [[-1]])
        >>> f = lambda t: [math.sin(t)]
        >>> x0 = [1]
        >>> x_t = solve_linear_ode_system_nonhomogeneous(A, f, x0, 2.0)
    """
    if not A.is_square():
        from ..error import NotSquareError
        raise NotSquareError((A.rows, A.cols), "ODE system solve")
    
    n = A.rows
    x = initial_conditions[:]
    current_t = 0.0
    
    # Runge-Kutta 4th order
    while current_t < t:
        # Adjust step size if needed
        step = min(dt, t - current_t)
        
        # k1 = f(t, x)
        k1 = _ode_derivative(A, f, current_t, x)
        
        # k2 = f(t + h/2, x + h*k1/2)
        x_temp = [x[i] + step * k1[i] / 2 for i in range(n)]
        k2 = _ode_derivative(A, f, current_t + step / 2, x_temp)
        
        # k3 = f(t + h/2, x + h*k2/2)
        x_temp = [x[i] + step * k2[i] / 2 for i in range(n)]
        k3 = _ode_derivative(A, f, current_t + step / 2, x_temp)
        
        # k4 = f(t + h, x + h*k3)
        x_temp = [x[i] + step * k3[i] for i in range(n)]
        k4 = _ode_derivative(A, f, current_t + step, x_temp)
        
        # Update x
        for i in range(n):
            x[i] += step * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
        
        current_t += step
    
    return x


def _ode_derivative(A: Matrix, f: Callable[[float], List[float]], t: float, x: List[float]) -> List[float]:
    """Helper: Calculate dx/dt = Ax + f(t)"""
    n = len(x)
    result = [0.0] * n
    
    # Ax
    for i in range(n):
        for j in range(n):
            result[i] += A.data[i][j] * x[j]
    
    # + f(t)
    f_t = f(t)
    for i in range(n):
        result[i] += f_t[i]
    
    return result


def euler_method(
    f: Callable[[float, List[float]], List[float]],
    initial_conditions: List[float],
    t0: float,
    t_final: float,
    dt: float = 0.01
) -> Tuple[List[float], List[List[float]]]:
    """
    Solve ODE system dx/dt = f(t, x) using Euler's method.
    
    Args:
        f: Function f(t, x) returning derivative vector
        initial_conditions: Initial values x(t0)
        t0: Initial time
        t_final: Final time
        dt: Time step
    
    Returns:
        Tuple (time_points, solution_vectors)
    
    Example:
        >>> # Solve: dx/dt = -x, dy/dt = -y
        >>> f = lambda t, x: [-x[0], -x[1]]
        >>> times, solutions = euler_method(f, [1, 1], 0, 2, 0.1)
    """
    t_values = []
    x_values = []
    
    t = t0
    x = initial_conditions[:]
    
    while t <= t_final:
        t_values.append(t)
        x_values.append(x[:])
        
        # Euler step: x_{n+1} = x_n + dt * f(t_n, x_n)
        derivative = f(t, x)
        x = [x[i] + dt * derivative[i] for i in range(len(x))]
        t += dt
    
    return t_values, x_values


def runge_kutta_4(
    f: Callable[[float, List[float]], List[float]],
    initial_conditions: List[float],
    t0: float,
    t_final: float,
    dt: float = 0.01
) -> Tuple[List[float], List[List[float]]]:
    """
    Solve ODE system dx/dt = f(t, x) using 4th-order Runge-Kutta method.
    
    Args:
        f: Function f(t, x) returning derivative vector
        initial_conditions: Initial values x(t0)
        t0: Initial time
        t_final: Final time
        dt: Time step
    
    Returns:
        Tuple (time_points, solution_vectors)
    
    Example:
        >>> # Solve: dx/dt = y, dy/dt = -x (harmonic oscillator)
        >>> f = lambda t, x: [x[1], -x[0]]
        >>> times, solutions = runge_kutta_4(f, [1, 0], 0, 10, 0.1)
    """
    t_values = []
    x_values = []
    
    t = t0
    x = initial_conditions[:]
    n = len(x)
    
    while t <= t_final:
        t_values.append(t)
        x_values.append(x[:])
        
        # RK4 steps
        k1 = f(t, x)
        
        x_temp = [x[i] + dt * k1[i] / 2 for i in range(n)]
        k2 = f(t + dt / 2, x_temp)
        
        x_temp = [x[i] + dt * k2[i] / 2 for i in range(n)]
        k3 = f(t + dt / 2, x_temp)
        
        x_temp = [x[i] + dt * k3[i] for i in range(n)]
        k4 = f(t + dt, x_temp)
        
        # Update
        for i in range(n):
            x[i] += dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
        
        t += dt
    
    return t_values, x_values
