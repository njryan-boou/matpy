"""
Test suite for matrix solve module (linear systems and differential equations).
"""

import unittest
import math
from src.matpy.matrix.core import Matrix
from src.matpy.matrix.solve import *
from src.matpy.error import NotSquareError, SingularMatrixError


class TestLinearSystems(unittest.TestCase):
    """Test linear system solvers."""
    
    def test_solve_linear_system_2x2(self):
        """Test solving 2x2 system."""
        # 2x + y = 5
        # x + 3y = 7
        # Solution: x=1, y=3
        A = Matrix(2, 2, [[2, 1], [1, 3]])
        b = [5, 7]
        x = solve_linear_system(A, b)
        self.assertAlmostEqual(x[0], 1, places=10)
        self.assertAlmostEqual(x[1], 2, places=10)
    
    def test_solve_linear_system_3x3(self):
        """Test solving 3x3 system."""
        A = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b = [8, -11, -3]
        x = solve_linear_system(A, b)
        # Verify solution
        result = [sum(A.data[i][j] * x[j] for j in range(3)) for i in range(3)]
        for i in range(3):
            self.assertAlmostEqual(result[i], b[i], places=8)
    
    def test_solve_singular_system(self):
        """Test that solving singular system raises error."""
        A = Matrix(2, 2, [[1, 2], [2, 4]])  # Singular
        b = [1, 2]
        with self.assertRaises(SingularMatrixError):
            solve_linear_system(A, b)
    
    def test_lu_decomposition(self):
        """Test LU decomposition."""
        A = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        L, U = lu_decomposition(A)
        # Verify A = LU
        product = L @ U
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(product.data[i][j], A.data[i][j], places=10)
    
    def test_solve_lu(self):
        """Test solving with LU decomposition."""
        A = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b = [8, -11, -3]
        x = solve_lu(A, b)
        # Verify solution
        result = [sum(A.data[i][j] * x[j] for j in range(3)) for i in range(3)]
        for i in range(3):
            self.assertAlmostEqual(result[i], b[i], places=8)
    
    def test_solve_cramer(self):
        """Test Cramer's rule."""
        A = Matrix(2, 2, [[1, 2], [3, 4]])
        b = [5, 11]
        x = solve_cramer(A, b)
        self.assertAlmostEqual(x[0], 1, places=10)
        self.assertAlmostEqual(x[1], 2, places=10)
    
    def test_solve_least_squares(self):
        """Test least squares solution."""
        # Overdetermined system (4 equations, 2 unknowns)
        # Fit line y = mx + c to points (0,1), (1,2), (2,4), (3,5)
        A = Matrix(4, 2, [[0, 1], [1, 1], [2, 1], [3, 1]])
        b = [1, 2, 4, 5]
        x = solve_least_squares(A, b)
        # Should get approximately m≈1.4, c≈0.5
        self.assertAlmostEqual(x[0], 1.4, places=1)
        self.assertAlmostEqual(x[1], 0.5, places=1)


class TestMatrixExponential(unittest.TestCase):
    """Test matrix exponential."""
    
    def test_matrix_exponential_identity(self):
        """Test exp(0) = I."""
        A = Matrix(2, 2, [[0, 0], [0, 0]])
        exp_A = matrix_exponential(A, 1.0)
        # Should be identity
        self.assertAlmostEqual(exp_A.data[0][0], 1, places=5)
        self.assertAlmostEqual(exp_A.data[0][1], 0, places=5)
        self.assertAlmostEqual(exp_A.data[1][0], 0, places=5)
        self.assertAlmostEqual(exp_A.data[1][1], 1, places=5)
    
    def test_matrix_exponential_diagonal(self):
        """Test exp of diagonal matrix."""
        A = Matrix(2, 2, [[1, 0], [0, 2]])
        exp_A = matrix_exponential(A, 1.0)
        # Diagonal should be exp(1), exp(2)
        self.assertAlmostEqual(exp_A.data[0][0], math.e, places=3)
        self.assertAlmostEqual(exp_A.data[1][1], math.e**2, places=3)


class TestODESystems(unittest.TestCase):
    """Test ODE system solvers."""
    
    def test_solve_homogeneous_1x1(self):
        """Test 1x1 homogeneous ODE: dx/dt = -2x, x(0)=1."""
        A = Matrix(1, 1, [[-2]])
        x0 = [1]
        t = 1.0
        x_t = solve_linear_ode_system_homogeneous(A, x0, t)
        # Solution: x(t) = exp(-2t) * x0
        expected = math.exp(-2 * t)
        self.assertAlmostEqual(x_t[0], expected, places=5)
    
    def test_solve_homogeneous_2x2(self):
        """Test 2x2 homogeneous ODE system."""
        # Simple diagonal system
        A = Matrix(2, 2, [[-1, 0], [0, -2]])
        x0 = [1, 1]
        t = 1.0
        x_t = solve_linear_ode_system_homogeneous(A, x0, t)
        # Solution: x1(t) = exp(-t), x2(t) = exp(-2t)
        self.assertAlmostEqual(x_t[0], math.exp(-t), places=3)
        self.assertAlmostEqual(x_t[1], math.exp(-2*t), places=3)
    
    def test_solve_nonhomogeneous(self):
        """Test non-homogeneous ODE system."""
        A = Matrix(1, 1, [[-1]])
        f = lambda t: [0]  # Homogeneous for simplicity
        x0 = [1]
        t = 1.0
        x_t = solve_linear_ode_system_nonhomogeneous(A, f, x0, t, dt=0.01)
        # Should approximate exp(-t)
        expected = math.exp(-t)
        self.assertAlmostEqual(x_t[0], expected, places=2)
    
    def test_euler_method(self):
        """Test Euler's method."""
        # Simple ODE: dx/dt = -x, x(0) = 1
        f = lambda t, x: [-x[0]]
        times, solutions = euler_method(f, [1], 0, 1, 0.1)
        # Final solution should approximate exp(-1)
        self.assertAlmostEqual(solutions[-1][0], math.exp(-1), places=1)
    
    def test_runge_kutta_4(self):
        """Test RK4 method."""
        # Simple ODE: dx/dt = -x, x(0) = 1
        f = lambda t, x: [-x[0]]
        times, solutions = runge_kutta_4(f, [1], 0, 1, 0.1)
        # RK4 should be more accurate than Euler
        self.assertAlmostEqual(solutions[-1][0], math.exp(-1), places=3)


if __name__ == '__main__':
    unittest.main()
