"""
Matrix Examples - Comprehensive demonstration of Matrix class capabilities.

This file showcases:
- Matrix creation and initialization
- Arithmetic operations
- Matrix-specific operations (transpose, determinant, inverse, etc.)
- Linear system solving
- Advanced operations
- Coordinate transformations
"""

import math
from src.matpy.matrix.core import Matrix
from src.matpy.matrix import ops
from src.matpy.matrix import solve


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def example_creation():
    """Demonstrate matrix creation methods."""
    print_section("Matrix Creation")
    
    # Zero matrix
    print("Zero Matrix (3x4):")
    m1 = Matrix(3, 4)
    print(m1)
    
    # Matrix with data
    print("\nMatrix with initial data (2x3):")
    m2 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
    print(m2)
    
    # Identity-like matrix
    print("\n2x2 Matrix:")
    m3 = Matrix(2, 2, [[1, 2], [3, 4]])
    print(m3)
    
    # Using ops module
    print("\nIdentity Matrix (3x3):")
    identity = ops.identity(3)
    print(identity)
    
    print("\nMatrix of ones (2x3):")
    ones = ops.ones(2, 3)
    print(ones)
    
    print("\nDiagonal Matrix:")
    diag = ops.diagonal([1, 2, 3, 4])
    print(diag)
    
    print("\nFrom rows:")
    from_rows = ops.from_rows([[1, 2], [3, 4], [5, 6]])
    print(from_rows)
    
    print("\nFrom columns:")
    from_cols = ops.from_columns([[1, 3, 5], [2, 4, 6]])
    print(from_cols)


def example_arithmetic():
    """Demonstrate arithmetic operations."""
    print_section("Arithmetic Operations")
    
    A = Matrix(2, 2, [[1, 2], [3, 4]])
    B = Matrix(2, 2, [[5, 6], [7, 8]])
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    
    print("\nA + B:")
    print(A + B)
    
    print("\nA - B:")
    print(A - B)
    
    print("\nA * 2 (scalar multiplication):")
    print(A * 2)
    
    print("\nA / 2 (scalar division):")
    print(A / 2)
    
    print("\n-A (negation):")
    print(-A)
    
    print("\nA @ B (matrix multiplication):")
    print(A @ B)
    
    print("\nA ** 2 (matrix power):")
    print(A ** 2)
    
    print("\nA ** 0 (should be identity):")
    print(A ** 0)


def example_matrix_operations():
    """Demonstrate matrix-specific operations."""
    print_section("Matrix Operations")
    
    A = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Matrix A (3x3):")
    print(A)
    
    print("\nTranspose:")
    print(A.transpose())
    
    print("\nTrace (sum of diagonal):")
    print(f"trace(A) = {A.trace()}")
    
    print("\nDeterminant:")
    print(f"det(A) = {A.determinant()}")
    
    print("\nRank:")
    print(f"rank(A) = {A.rank()}")
    
    print("\nFrobenius Norm (abs):")
    print(f"|A| = {abs(A):.4f}")
    
    # Invertible matrix
    B = Matrix(3, 3, [[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    print("\n\nMatrix B (invertible):")
    print(B)
    
    print("\nInverse of B:")
    B_inv = B.inverse()
    print(B_inv)
    
    print("\nB @ B^(-1) (should be identity):")
    print(B @ B_inv)
    
    print("\nCofactor matrix:")
    print(B.cofactor())
    
    print("\nAdjugate matrix:")
    print(B.adjugate())


def example_properties():
    """Demonstrate matrix property checks."""
    print_section("Matrix Properties")
    
    # Symmetric matrix
    S = Matrix(3, 3, [[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    print("Symmetric Matrix S:")
    print(S)
    print(f"Is square: {S.is_square()}")
    print(f"Is symmetric: {S.is_symmetric()}")
    print(f"Is singular: {S.is_singular()}")
    print(f"Is invertible: {S.is_invertible()}")
    
    # Identity matrix
    I = ops.identity(3)
    print("\n\nIdentity Matrix I:")
    print(I)
    print(f"Is identity: {ops.is_identity(I)}")
    print(f"Is diagonal: {ops.is_diagonal(I)}")
    
    # Upper triangular
    U = Matrix(3, 3, [[1, 2, 3], [0, 4, 5], [0, 0, 6]])
    print("\n\nUpper Triangular Matrix U:")
    print(U)
    print(f"Is upper triangular: {ops.is_upper_triangular(U)}")
    
    # Lower triangular
    L = Matrix(3, 3, [[1, 0, 0], [2, 3, 0], [4, 5, 6]])
    print("\n\nLower Triangular Matrix L:")
    print(L)
    print(f"Is lower triangular: {ops.is_lower_triangular(L)}")


def example_linear_systems():
    """Demonstrate solving linear systems."""
    print_section("Solving Linear Systems")
    
    print("System of equations:")
    print("  2x + y = 5")
    print("  x + 3y = 7")
    
    A = Matrix(2, 2, [[2, 1], [1, 3]])
    b = [5, 7]
    
    print("\nCoefficient matrix A:")
    print(A)
    print(f"\nRight-hand side b: {b}")
    
    # Gaussian elimination
    print("\nSolution using Gaussian elimination:")
    x = solve.solve_linear_system(A, b)
    print(f"x = {x}")
    
    # Cramer's rule
    print("\nSolution using Cramer's rule:")
    x_cramer = solve.solve_cramer(A, b)
    print(f"x = {x_cramer}")
    
    # LU decomposition
    print("\nSolution using LU decomposition:")
    x_lu = solve.solve_lu(A, b)
    print(f"x = {x_lu}")
    
    # Larger system
    print("\n\nLarger system (3x3):")
    A3 = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    b3 = [8, -11, -3]
    x3 = solve.solve_linear_system(A3, b3)
    print(f"Solution: {x3}")
    
    # Verify solution
    result = [sum(A3.data[i][j] * x3[j] for j in range(3)) for i in range(3)]
    print(f"Verification Ax = {result}")
    print(f"Should equal b = {b3}")


def example_least_squares():
    """Demonstrate least squares fitting."""
    print_section("Least Squares Fitting")
    
    print("Fit line y = mx + c to points:")
    points = [(0, 1), (1, 2), (2, 4), (3, 5)]
    for x, y in points:
        print(f"  ({x}, {y})")
    
    # Create overdetermined system
    A = Matrix(4, 2, [[x, 1] for x, y in points])
    b = [y for x, y in points]
    
    print("\nDesign matrix A:")
    print(A)
    print(f"\nObservations b: {b}")
    
    x = solve.solve_least_squares(A, b)
    print(f"\nLeast squares solution: m = {x[0]:.4f}, c = {x[1]:.4f}")
    print(f"Line equation: y = {x[0]:.4f}x + {x[1]:.4f}")


def example_advanced_operations():
    """Demonstrate advanced operations."""
    print_section("Advanced Operations")
    
    A = Matrix(2, 2, [[1, 2], [3, 4]])
    B = Matrix(2, 2, [[5, 6], [7, 8]])
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    
    # Hadamard product
    print("\nHadamard product (element-wise multiplication):")
    hadamard = ops.hadamard_product(A, B)
    print(hadamard)
    
    # Kronecker product
    print("\nKronecker product:")
    kronecker = ops.kronecker_product(A, B)
    print(kronecker)
    
    # Row echelon form
    C = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    print("\n\nMatrix C:")
    print(C)
    
    print("\nRow echelon form:")
    ref = ops.row_echelon_form(C)
    print(ref)
    
    print("\nReduced row echelon form:")
    rref = ops.reduced_row_echelon_form(C)
    print(rref)
    
    # Concatenation
    print("\n\nHorizontal concatenation [A | B]:")
    h_concat = ops.concatenate_horizontal(A, B)
    print(h_concat)
    
    print("\nVertical concatenation [A; B]:")
    v_concat = ops.concatenate_vertical(A, B)
    print(v_concat)


def example_differential_equations():
    """Demonstrate ODE system solving."""
    print_section("Differential Equations")
    
    print("Solving ODE system: dx/dt = Ax")
    print("\nExample 1: dx/dt = -2x, x(0) = 1")
    
    A = Matrix(1, 1, [[-2]])
    x0 = [1]
    t = 1.0
    
    x_t = solve.solve_linear_ode_system_homogeneous(A, x0, t)
    exact = math.exp(-2 * t)
    print(f"  x({t}) = {x_t[0]:.6f}")
    print(f"  Exact: {exact:.6f}")
    
    print("\n\nExample 2: 2D system")
    print("  dx/dt = -x")
    print("  dy/dt = -2y")
    print("  x(0) = 1, y(0) = 1")
    
    A2 = Matrix(2, 2, [[-1, 0], [0, -2]])
    x0_2 = [1, 1]
    t = 1.0
    
    x_t2 = solve.solve_linear_ode_system_homogeneous(A2, x0_2, t)
    print(f"  x({t}) = {x_t2[0]:.6f}")
    print(f"  y({t}) = {x_t2[1]:.6f}")
    print(f"  Exact: x={math.exp(-t):.6f}, y={math.exp(-2*t):.6f}")
    
    print("\n\nUsing Runge-Kutta 4:")
    print("  Solving: dx/dt = -x, x(0) = 1")
    f = lambda t, x: [-x[0]]
    times, solutions = solve.runge_kutta_4(f, [1], 0, 1, 0.1)
    print(f"  x(1) ≈ {solutions[-1][0]:.6f}")
    print(f"  Exact: {math.exp(-1):.6f}")


def example_indexing():
    """Demonstrate indexing and iteration."""
    print_section("Indexing and Iteration")
    
    A = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Matrix A:")
    print(A)
    
    print(f"\nA[0, 0] = {A[0, 0]}")
    print(f"A[1, 2] = {A[1, 2]}")
    print(f"A[2, 1] = {A[2, 1]}")
    
    print("\nSetting A[1, 1] = 99:")
    A[1, 1] = 99
    print(A)
    
    print("\nIterating over rows:")
    for i, row in enumerate(A):
        print(f"  Row {i}: {row}")
    
    print(f"\nIs 99 in matrix? {99 in A}")
    print(f"Is 100 in matrix? {100 in A}")
    
    print(f"\nMatrix length (number of rows): {len(A)}")


def example_comparison():
    """Demonstrate comparison operations."""
    print_section("Comparison Operations")
    
    A = Matrix(2, 2, [[1, 2], [3, 4]])
    B = Matrix(2, 2, [[1, 2], [3, 4]])
    C = Matrix(2, 2, [[1, 2], [3, 5]])
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B (same as A):")
    print(B)
    print("\nMatrix C (different):")
    print(C)
    
    print(f"\nA == B: {A == B}")
    print(f"A == C: {A == C}")
    print(f"A != C: {A != C}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("  MATPY - MATRIX OPERATIONS EXAMPLES")
    print("="*60)
    
    example_creation()
    example_arithmetic()
    example_matrix_operations()
    example_properties()
    example_linear_systems()
    example_least_squares()
    example_advanced_operations()
    example_differential_equations()
    example_indexing()
    example_comparison()
    
    print("\n" + "="*60)
    print("  END OF EXAMPLES")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
