"""
Test suite for matrix operations module (ops.py).

Tests all standalone matrix operation functions.
"""

import unittest
import math
from src.matpy.matrix.core import Matrix
from src.matpy.matrix.ops import *
from src.matpy.error import ShapeError, NotSquareError


class TestMatrixCreation(unittest.TestCase):
    """Test matrix creation functions."""
    
    def test_zeros(self):
        """Test creating zero matrix."""
        m = zeros(3, 4)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 4)
        for i in range(3):
            for j in range(4):
                self.assertEqual(m.data[i][j], 0)
    
    def test_ones(self):
        """Test creating matrix of ones."""
        m = ones(2, 3)
        for i in range(2):
            for j in range(3):
                self.assertEqual(m.data[i][j], 1)
    
    def test_identity(self):
        """Test creating identity matrix."""
        m = identity(3)
        self.assertTrue(is_identity(m))
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertEqual(m.data[i][j], 1)
                else:
                    self.assertEqual(m.data[i][j], 0)
    
    def test_diagonal(self):
        """Test creating diagonal matrix."""
        m = diagonal([1, 2, 3, 4])
        self.assertEqual(m.data[0][0], 1)
        self.assertEqual(m.data[1][1], 2)
        self.assertEqual(m.data[2][2], 3)
        self.assertEqual(m.data[3][3], 4)
        self.assertEqual(m.data[0][1], 0)
    
    def test_from_rows(self):
        """Test creating matrix from rows."""
        rows = [[1, 2, 3], [4, 5, 6]]
        m = from_rows(rows)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m.data, rows)
    
    def test_from_columns(self):
        """Test creating matrix from columns."""
        cols = [[1, 4], [2, 5], [3, 6]]
        m = from_columns(cols)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 3)
        self.assertEqual(m.data, [[1, 2, 3], [4, 5, 6]])


class TestBasicOperations(unittest.TestCase):
    """Test basic matrix operations."""
    
    def test_transpose(self):
        """Test transpose function."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        t = transpose(m)
        self.assertEqual(t.data, [[1, 4], [2, 5], [3, 6]])
    
    def test_trace(self):
        """Test trace function."""
        m = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(trace(m), 15)
    
    def test_determinant(self):
        """Test determinant function."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        self.assertEqual(determinant(m), -2)
    
    def test_inverse(self):
        """Test inverse function."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        inv = inverse(m)
        identity_result = m @ inv
        self.assertAlmostEqual(identity_result.data[0][0], 1, places=10)
    
    def test_rank(self):
        """Test rank function."""
        m = Matrix(2, 2, [[1, 0], [0, 1]])
        self.assertEqual(rank(m), 2)


class TestMatrixProperties(unittest.TestCase):
    """Test matrix property checking functions."""
    
    def test_is_square(self):
        """Test is_square function."""
        m1 = Matrix(3, 3)
        m2 = Matrix(2, 3)
        self.assertTrue(is_square(m1))
        self.assertFalse(is_square(m2))
    
    def test_is_symmetric(self):
        """Test is_symmetric function."""
        m1 = Matrix(3, 3, [[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        m2 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(is_symmetric(m1))
        self.assertFalse(is_symmetric(m2))
    
    def test_is_diagonal(self):
        """Test is_diagonal function."""
        m1 = diagonal([1, 2, 3])
        m2 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(is_diagonal(m1))
        self.assertFalse(is_diagonal(m2))
    
    def test_is_identity(self):
        """Test is_identity function."""
        m1 = identity(3)
        m2 = Matrix(3, 3, [[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.assertTrue(is_identity(m1))
        self.assertFalse(is_identity(m2))
    
    def test_is_upper_triangular(self):
        """Test is_upper_triangular function."""
        m1 = Matrix(3, 3, [[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        m2 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(is_upper_triangular(m1))
        self.assertFalse(is_upper_triangular(m2))
    
    def test_is_lower_triangular(self):
        """Test is_lower_triangular function."""
        m1 = Matrix(3, 3, [[1, 0, 0], [2, 3, 0], [4, 5, 6]])
        m2 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(is_lower_triangular(m1))
        self.assertFalse(is_lower_triangular(m2))


class TestAdvancedOperations(unittest.TestCase):
    """Test advanced matrix operations."""
    
    def test_power(self):
        """Test matrix power function."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = power(m, 2)
        self.assertEqual(m2.data, [[7, 10], [15, 22]])
    
    def test_hadamard_product(self):
        """Test Hadamard (element-wise) product."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[5, 6], [7, 8]])
        result = hadamard_product(m1, m2)
        self.assertEqual(result.data, [[5, 12], [21, 32]])
    
    def test_hadamard_dimension_mismatch(self):
        """Test that Hadamard product with wrong dimensions raises error."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ShapeError):
            hadamard_product(m1, m2)
    
    def test_kronecker_product(self):
        """Test Kronecker product."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[0, 5], [6, 7]])
        result = kronecker_product(m1, m2)
        self.assertEqual(result.rows, 4)
        self.assertEqual(result.cols, 4)
    
    def test_frobenius_norm(self):
        """Test Frobenius norm."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        norm = frobenius_norm(m)
        expected = math.sqrt(1 + 4 + 9 + 16)
        self.assertAlmostEqual(norm, expected, places=10)
    
    def test_row_echelon_form(self):
        """Test row echelon form."""
        m = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        ref = row_echelon_form(m)
        # Check that it's in row echelon form (leading entries)
        # This is a basic check - full verification would be more complex
        self.assertIsInstance(ref, Matrix)
    
    def test_reduced_row_echelon_form(self):
        """Test reduced row echelon form."""
        m = Matrix(3, 3, [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        rref = reduced_row_echelon_form(m)
        self.assertIsInstance(rref, Matrix)


class TestConcatenation(unittest.TestCase):
    """Test matrix concatenation."""
    
    def test_concatenate_horizontal(self):
        """Test horizontal concatenation."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 3, [[5, 6, 7], [8, 9, 10]])
        result = concatenate_horizontal(m1, m2)
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.cols, 5)
        self.assertEqual(result.data, [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]])
    
    def test_concatenate_horizontal_dimension_mismatch(self):
        """Test that horizontal concatenation with wrong rows raises error."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(3, 2, [[5, 6], [7, 8], [9, 10]])
        with self.assertRaises(ShapeError):
            concatenate_horizontal(m1, m2)
    
    def test_concatenate_vertical(self):
        """Test vertical concatenation."""
        m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        m2 = Matrix(2, 3, [[7, 8, 9], [10, 11, 12]])
        result = concatenate_vertical(m1, m2)
        self.assertEqual(result.rows, 4)
        self.assertEqual(result.cols, 3)
        self.assertEqual(result.data, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    
    def test_concatenate_vertical_dimension_mismatch(self):
        """Test that vertical concatenation with wrong cols raises error."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 3, [[5, 6, 7], [8, 9, 10]])
        with self.assertRaises(ShapeError):
            concatenate_vertical(m1, m2)


if __name__ == '__main__':
    unittest.main()
