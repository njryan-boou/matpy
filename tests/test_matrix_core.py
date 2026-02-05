"""
Comprehensive test suite for Matrix class.

Tests all aspects of the Matrix class including:
- Initialization and properties
- String representation
- Arithmetic operations
- Matrix-specific operations (transpose, determinant, inverse, etc.)
- Dunder methods
- Edge cases and error handling
"""

import unittest
import math
from src.matpy.matrix.core import Matrix
from src.matpy.error import (
    ValidationError, ShapeError, NotSquareError, 
    SingularMatrixError, MatPyIndexError
)


class TestMatrixInitialization(unittest.TestCase):
    """Test matrix initialization and basic properties."""
    
    def test_zero_matrix(self):
        """Test creating a zero matrix."""
        m = Matrix(2, 3)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 3)
        for i in range(2):
            for j in range(3):
                self.assertEqual(m.data[i][j], 0.0)
    
    def test_matrix_with_data(self):
        """Test creating matrix with initial data."""
        data = [[1, 2, 3], [4, 5, 6]]
        m = Matrix(2, 3, data)
        self.assertEqual(m.data, data)
    
    def test_identity_like(self):
        """Test creating identity-like matrix."""
        data = [[1, 0], [0, 1]]
        m = Matrix(2, 2, data)
        self.assertTrue(m.is_square())
        self.assertEqual(m.data[0][0], 1)
        self.assertEqual(m.data[1][1], 1)
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        with self.assertRaises(ValidationError):
            Matrix(0, 3)
        with self.assertRaises(ValidationError):
            Matrix(2, -1)
    
    def test_data_shape_mismatch(self):
        """Test that data shape mismatch raises error."""
        data = [[1, 2], [3, 4, 5]]  # Inconsistent row lengths
        with self.assertRaises(ValidationError):
            Matrix(2, 2, data)


class TestMatrixStringRepresentation(unittest.TestCase):
    """Test string representation methods."""
    
    def test_str(self):
        """Test __str__ method."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        s = str(m)
        self.assertIn("1", s)
        self.assertIn("2", s)
        self.assertIn("3", s)
        self.assertIn("4", s)
    
    def test_repr(self):
        """Test __repr__ method."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        r = repr(m)
        self.assertIn("Matrix", r)
        self.assertIn("2x2", r)


class TestMatrixArithmetic(unittest.TestCase):
    """Test arithmetic operations."""
    
    def test_addition(self):
        """Test matrix addition."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[5, 6], [7, 8]])
        result = m1 + m2
        self.assertEqual(result.data, [[6, 8], [10, 12]])
    
    def test_addition_dimension_mismatch(self):
        """Test that adding matrices of different sizes raises error."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ShapeError):
            m1 + m2
    
    def test_subtraction(self):
        """Test matrix subtraction."""
        m1 = Matrix(2, 2, [[5, 6], [7, 8]])
        m2 = Matrix(2, 2, [[1, 2], [3, 4]])
        result = m1 - m2
        self.assertEqual(result.data, [[4, 4], [4, 4]])
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = m * 2
        self.assertEqual(result.data, [[2, 4], [6, 8]])
    
    def test_scalar_multiplication_reflected(self):
        """Test reflected scalar multiplication."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = 2 * m
        self.assertEqual(result.data, [[2, 4], [6, 8]])
    
    def test_scalar_division(self):
        """Test scalar division."""
        m = Matrix(2, 2, [[2, 4], [6, 8]])
        result = m / 2
        self.assertEqual(result.data, [[1, 2], [3, 4]])
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        m2 = Matrix(3, 2, [[7, 8], [9, 10], [11, 12]])
        result = m1 @ m2
        # [1*7+2*9+3*11, 1*8+2*10+3*12]   [58, 64]
        # [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        self.assertEqual(result.data, [[58, 64], [139, 154]])
    
    def test_matrix_multiplication_incompatible(self):
        """Test that incompatible matrix multiplication raises error."""
        m1 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        m2 = Matrix(2, 2, [[1, 2], [3, 4]])
        with self.assertRaises(ShapeError):
            m1 @ m2
    
    def test_negation(self):
        """Test matrix negation."""
        m = Matrix(2, 2, [[1, -2], [-3, 4]])
        result = -m
        self.assertEqual(result.data, [[-1, 2], [3, -4]])
    
    def test_in_place_addition(self):
        """Test in-place addition."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[5, 6], [7, 8]])
        m1 += m2
        self.assertEqual(m1.data, [[6, 8], [10, 12]])
    
    def test_in_place_subtraction(self):
        """Test in-place subtraction."""
        m1 = Matrix(2, 2, [[5, 6], [7, 8]])
        m2 = Matrix(2, 2, [[1, 2], [3, 4]])
        m1 -= m2
        self.assertEqual(m1.data, [[4, 4], [4, 4]])
    
    def test_in_place_scalar_multiplication(self):
        """Test in-place scalar multiplication."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        m *= 2
        self.assertEqual(m.data, [[2, 4], [6, 8]])


class TestMatrixPower(unittest.TestCase):
    """Test matrix power operations."""
    
    def test_power_zero(self):
        """Test matrix to power 0 (should return identity)."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = m ** 0
        self.assertEqual(result.data, [[1, 0], [0, 1]])
    
    def test_power_one(self):
        """Test matrix to power 1."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = m ** 1
        self.assertEqual(result.data, [[1, 2], [3, 4]])
    
    def test_power_two(self):
        """Test matrix squared."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = m ** 2
        # [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
        self.assertEqual(result.data, [[7, 10], [15, 22]])
    
    def test_power_non_square(self):
        """Test that power of non-square matrix raises error."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(NotSquareError):
            m ** 2


class TestMatrixComparison(unittest.TestCase):
    """Test comparison operators."""
    
    def test_equality(self):
        """Test matrix equality."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[1, 2], [3, 4]])
        self.assertTrue(m1 == m2)
    
    def test_inequality(self):
        """Test matrix inequality."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 2, [[1, 2], [3, 5]])
        self.assertTrue(m1 != m2)
    
    def test_equality_different_sizes(self):
        """Test that different size matrices are not equal."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        self.assertFalse(m1 == m2)


class TestMatrixIndexing(unittest.TestCase):
    """Test indexing and item access."""
    
    def test_getitem(self):
        """Test getting items by index."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[0, 1], 2)
        self.assertEqual(m[1, 0], 3)
        self.assertEqual(m[1, 1], 4)
    
    def test_setitem(self):
        """Test setting items by index."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        m[0, 0] = 10
        m[1, 1] = 20
        self.assertEqual(m[0, 0], 10)
        self.assertEqual(m[1, 1], 20)
    
    def test_getitem_out_of_range(self):
        """Test that out of range access raises error."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        with self.assertRaises(MatPyIndexError):
            _ = m[5, 0]
    
    def test_iteration(self):
        """Test iterating over rows."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        rows = list(m)
        self.assertEqual(rows, [[1, 2], [3, 4]])
    
    def test_contains(self):
        """Test membership testing."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        self.assertTrue(2 in m)
        self.assertFalse(10 in m)


class TestMatrixTranspose(unittest.TestCase):
    """Test transpose operation."""
    
    def test_transpose_square(self):
        """Test transposing a square matrix."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        result = m.transpose()
        self.assertEqual(result.data, [[1, 3], [2, 4]])
    
    def test_transpose_rectangular(self):
        """Test transposing a rectangular matrix."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        result = m.transpose()
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.cols, 2)
        self.assertEqual(result.data, [[1, 4], [2, 5], [3, 6]])
    
    def test_transpose_twice(self):
        """Test that transposing twice gives original."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        result = m.transpose().transpose()
        self.assertEqual(result.data, m.data)


class TestMatrixDeterminant(unittest.TestCase):
    """Test determinant calculation."""
    
    def test_determinant_1x1(self):
        """Test determinant of 1x1 matrix."""
        m = Matrix(1, 1, [[5]])
        self.assertEqual(m.determinant(), 5)
    
    def test_determinant_2x2(self):
        """Test determinant of 2x2 matrix."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        # det = 1*4 - 2*3 = -2
        self.assertEqual(m.determinant(), -2)
    
    def test_determinant_3x3(self):
        """Test determinant of 3x3 matrix."""
        m = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # This matrix is singular (rows are linearly dependent)
        self.assertAlmostEqual(m.determinant(), 0, places=10)
    
    def test_determinant_identity(self):
        """Test determinant of identity matrix."""
        m = Matrix(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(m.determinant(), 1)
    
    def test_determinant_non_square(self):
        """Test that determinant of non-square matrix raises error."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(NotSquareError):
            m.determinant()


class TestMatrixInverse(unittest.TestCase):
    """Test matrix inversion."""
    
    def test_inverse_2x2(self):
        """Test inverting a 2x2 matrix."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        inv = m.inverse()
        # Verify M * M^-1 = I
        identity = m @ inv
        self.assertAlmostEqual(identity.data[0][0], 1, places=10)
        self.assertAlmostEqual(identity.data[0][1], 0, places=10)
        self.assertAlmostEqual(identity.data[1][0], 0, places=10)
        self.assertAlmostEqual(identity.data[1][1], 1, places=10)
    
    def test_inverse_3x3(self):
        """Test inverting a 3x3 matrix."""
        m = Matrix(3, 3, [[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        inv = m.inverse()
        identity = m @ inv
        # Check diagonal is ~1
        for i in range(3):
            self.assertAlmostEqual(identity.data[i][i], 1, places=10)
    
    def test_inverse_singular(self):
        """Test that inverting singular matrix raises error."""
        m = Matrix(2, 2, [[1, 2], [2, 4]])  # Singular (det=0)
        with self.assertRaises(SingularMatrixError):
            m.inverse()
    
    def test_inverse_non_square(self):
        """Test that inverse of non-square matrix raises error."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(NotSquareError):
            m.inverse()


class TestMatrixTrace(unittest.TestCase):
    """Test trace calculation."""
    
    def test_trace(self):
        """Test trace calculation."""
        m = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # trace = 1 + 5 + 9 = 15
        self.assertEqual(m.trace(), 15)
    
    def test_trace_identity(self):
        """Test trace of identity matrix."""
        m = Matrix(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(m.trace(), 3)
    
    def test_trace_non_square(self):
        """Test that trace of non-square matrix raises error."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(NotSquareError):
            m.trace()


class TestMatrixRank(unittest.TestCase):
    """Test rank calculation."""
    
    def test_rank_full(self):
        """Test rank of full rank matrix."""
        m = Matrix(2, 2, [[1, 0], [0, 1]])
        self.assertEqual(m.rank(), 2)
    
    def test_rank_deficient(self):
        """Test rank of rank-deficient matrix."""
        m = Matrix(3, 3, [[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        # All rows are multiples of first row
        self.assertEqual(m.rank(), 1)
    
    def test_rank_zero(self):
        """Test rank of zero matrix."""
        m = Matrix(2, 2)
        self.assertEqual(m.rank(), 0)


class TestMatrixProperties(unittest.TestCase):
    """Test matrix property checks."""
    
    def test_is_square(self):
        """Test square matrix detection."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])
        m2 = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        self.assertTrue(m1.is_square())
        self.assertFalse(m2.is_square())
    
    def test_is_symmetric(self):
        """Test symmetric matrix detection."""
        m1 = Matrix(3, 3, [[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        m2 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(m1.is_symmetric())
        self.assertFalse(m2.is_symmetric())
    
    def test_is_singular(self):
        """Test singular matrix detection."""
        m1 = Matrix(2, 2, [[1, 2], [2, 4]])  # Singular
        m2 = Matrix(2, 2, [[1, 2], [3, 4]])  # Non-singular
        self.assertTrue(m1.is_singular())
        self.assertFalse(m2.is_singular())
    
    def test_is_invertible(self):
        """Test invertible matrix detection."""
        m1 = Matrix(2, 2, [[1, 2], [3, 4]])  # Invertible
        m2 = Matrix(2, 2, [[1, 2], [2, 4]])  # Not invertible
        self.assertTrue(m1.is_invertible())
        self.assertFalse(m2.is_invertible())


class TestMatrixCopying(unittest.TestCase):
    """Test copying operations."""
    
    def test_copy(self):
        """Test shallow copy."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        copy = m.__copy__()
        self.assertEqual(copy.data, m.data)
        # Modify copy
        copy.data[0][0] = 10
        # Original should be unchanged (data is deep copied)
        self.assertEqual(m.data[0][0], 1)
    
    def test_deepcopy(self):
        """Test deep copy."""
        import copy
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        deep = copy.deepcopy(m)
        self.assertEqual(deep.data, m.data)
        deep.data[0][0] = 10
        self.assertEqual(m.data[0][0], 1)


class TestMatrixAbs(unittest.TestCase):
    """Test absolute value (Frobenius norm)."""
    
    def test_abs(self):
        """Test Frobenius norm."""
        m = Matrix(2, 2, [[1, 2], [3, 4]])
        # sqrt(1 + 4 + 9 + 16) = sqrt(30)
        self.assertAlmostEqual(abs(m), math.sqrt(30), places=10)


if __name__ == '__main__':
    unittest.main()
