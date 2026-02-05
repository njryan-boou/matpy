"""
Unit tests for matpy.vector.core module.

Tests all Vector class methods including dunder methods, arithmetic operations,
and vector-specific operations.
"""

import unittest
import copy
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matpy.vector.core import Vector


class TestVectorInitialization(unittest.TestCase):
    """Test Vector initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test creating a vector with default values."""
        v = Vector()
        self.assertEqual(v.x, 0)
        self.assertEqual(v.y, 0)
        self.assertEqual(v.z, 0)
    
    def test_full_initialization(self):
        """Test creating a vector with all components."""
        v = Vector(1, 2, 3)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)
    
    def test_partial_initialization(self):
        """Test creating a vector with some components."""
        v = Vector(5, 10)
        self.assertEqual(v.x, 5)
        self.assertEqual(v.y, 10)
        self.assertEqual(v.z, 0)


class TestVectorStringRepresentation(unittest.TestCase):
    """Test string representation methods."""
    
    def test_str(self):
        """Test __str__ method."""
        v = Vector(1, 2, 3)
        self.assertEqual(str(v), "(1, 2, 3)")
    
    def test_repr(self):
        """Test __repr__ method."""
        v = Vector(1, 2, 3)
        self.assertEqual(repr(v), "<1, 2, 3>")
    
    def test_format(self):
        """Test __format__ method."""
        v = Vector(3.14159, 2.71828, 1.41421)
        formatted = f"{v:.2f}"
        self.assertEqual(formatted, "(3.14, 2.72, 1.41)")


class TestVectorArithmetic(unittest.TestCase):
    """Test arithmetic operations."""
    
    def test_addition(self):
        """Test vector addition."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        result = v1 + v2
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 7)
        self.assertEqual(result.z, 9)
    
    def test_reflected_addition(self):
        """Test reflected addition."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        result1 = v1 + v2
        result2 = v2 + v1
        self.assertEqual(result1.x, result2.x)
        self.assertEqual(result1.y, result2.y)
        self.assertEqual(result1.z, result2.z)
    
    def test_subtraction(self):
        """Test vector subtraction."""
        v1 = Vector(10, 8, 6)
        v2 = Vector(1, 2, 3)
        result = v1 - v2
        self.assertEqual(result.x, 9)
        self.assertEqual(result.y, 6)
        self.assertEqual(result.z, 3)
    
    def test_multiplication_by_scalar(self):
        """Test vector multiplication by scalar."""
        v = Vector(1, 2, 3)
        result = v * 3
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 6)
        self.assertEqual(result.z, 9)
    
    def test_reflected_multiplication(self):
        """Test reflected multiplication."""
        v = Vector(1, 2, 3)
        result = 3 * v
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 6)
        self.assertEqual(result.z, 9)
    
    def test_division_by_scalar(self):
        """Test vector division by scalar."""
        v = Vector(10, 20, 30)
        result = v / 2
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 10)
        self.assertEqual(result.z, 15)
    
    def test_division_by_zero(self):
        """Test division by zero raises error."""
        v = Vector(1, 2, 3)
        with self.assertRaises(ZeroDivisionError):
            result = v / 0


class TestVectorUnaryOperators(unittest.TestCase):
    """Test unary operators."""
    
    def test_negation(self):
        """Test vector negation."""
        v = Vector(1, -2, 3)
        result = -v
        self.assertEqual(result.x, -1)
        self.assertEqual(result.y, 2)
        self.assertEqual(result.z, -3)
    
    def test_positive(self):
        """Test unary plus."""
        v = Vector(1, 2, 3)
        result = +v
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 2)
        self.assertEqual(result.z, 3)
    
    def test_abs(self):
        """Test absolute value (magnitude)."""
        v = Vector(3, 4, 0)
        self.assertEqual(abs(v), 5.0)


class TestVectorTypeConversion(unittest.TestCase):
    """Test type conversion methods."""
    
    def test_bool_zero_vector(self):
        """Test bool conversion of zero vector."""
        v = Vector(0, 0, 0)
        self.assertFalse(bool(v))
    
    def test_bool_nonzero_vector(self):
        """Test bool conversion of non-zero vector."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        v3 = Vector(0, 0, 1)
        self.assertTrue(bool(v1))
        self.assertTrue(bool(v2))
        self.assertTrue(bool(v3))
    
    def test_round(self):
        """Test rounding vector components."""
        v = Vector(3.456, 2.789, 1.234)
        result = round(v, 1)
        self.assertEqual(result.x, 3.5)
        self.assertEqual(result.y, 2.8)
        self.assertEqual(result.z, 1.2)
    
    def test_round_default(self):
        """Test rounding to integers."""
        v = Vector(3.6, 2.4, 1.5)
        result = round(v)
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 2)
        self.assertEqual(result.z, 2)


class TestVectorSequenceMethods(unittest.TestCase):
    """Test sequence/container methods."""
    
    def test_len(self):
        """Test length of vector."""
        v = Vector(1, 2, 3)
        self.assertEqual(len(v), 3)
    
    def test_getitem(self):
        """Test indexing."""
        v = Vector(10, 20, 30)
        self.assertEqual(v[0], 10)
        self.assertEqual(v[1], 20)
        self.assertEqual(v[2], 30)
    
    def test_getitem_out_of_range(self):
        """Test indexing with invalid index."""
        v = Vector(1, 2, 3)
        with self.assertRaises(IndexError):
            _ = v[3]
        with self.assertRaises(IndexError):
            _ = v[-1]
    
    def test_iteration(self):
        """Test iterating over vector components."""
        v = Vector(1, 2, 3)
        components = list(v)
        self.assertEqual(components, [1, 2, 3])
    
    def test_contains(self):
        """Test membership testing."""
        v = Vector(1, 2, 3)
        self.assertTrue(1 in v)
        self.assertTrue(2 in v)
        self.assertTrue(3 in v)
        self.assertFalse(4 in v)


class TestVectorCopying(unittest.TestCase):
    """Test copying methods."""
    
    def test_copy(self):
        """Test shallow copy."""
        v1 = Vector(1, 2, 3)
        v2 = copy.copy(v1)
        
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        self.assertEqual(v1.z, v2.z)
        self.assertIsNot(v1, v2)


class TestVectorOperations(unittest.TestCase):
    """Test vector-specific operations."""
    
    def test_magnitude_zero(self):
        """Test magnitude of zero vector."""
        v = Vector(0, 0, 0)
        self.assertEqual(v.magnitude(), 0.0)
    
    def test_magnitude_unit_vectors(self):
        """Test magnitude of unit vectors."""
        vx = Vector(1, 0, 0)
        vy = Vector(0, 1, 0)
        vz = Vector(0, 0, 1)
        self.assertAlmostEqual(vx.magnitude(), 1.0)
        self.assertAlmostEqual(vy.magnitude(), 1.0)
        self.assertAlmostEqual(vz.magnitude(), 1.0)
    
    def test_magnitude_pythagorean(self):
        """Test magnitude using Pythagorean theorem."""
        v = Vector(3, 4, 0)
        self.assertAlmostEqual(v.magnitude(), 5.0)
    
    def test_dot_product(self):
        """Test dot product."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        result = v1.dot(v2)
        self.assertEqual(result, 32)  # 1*4 + 2*5 + 3*6 = 32
    
    def test_dot_product_orthogonal(self):
        """Test dot product of orthogonal vectors."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        result = v1.dot(v2)
        self.assertEqual(result, 0)
    
    def test_cross_product(self):
        """Test cross product."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        result = v1.cross(v2)
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 1)
    
    def test_cross_product_anticommutative(self):
        """Test cross product anticommutativity."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        result1 = v1.cross(v2)
        result2 = v2.cross(v1)
        self.assertEqual(result1.x, -result2.x)
        self.assertEqual(result1.y, -result2.y)
        self.assertEqual(result1.z, -result2.z)
    
    def test_normalize(self):
        """Test normalization."""
        v = Vector(3, 4, 0)
        result = v.normalize()
        self.assertAlmostEqual(result.magnitude(), 1.0)
        self.assertAlmostEqual(result.x, 0.6)
        self.assertAlmostEqual(result.y, 0.8)
        self.assertAlmostEqual(result.z, 0.0)
    
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        v = Vector(0, 0, 0)
        result = v.normalize()
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)


class TestVectorEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_negative_components(self):
        """Test vector with negative components."""
        v = Vector(-1, -2, -3)
        self.assertEqual(v.x, -1)
        self.assertEqual(v.y, -2)
        self.assertEqual(v.z, -3)
    
    def test_float_precision(self):
        """Test floating point precision."""
        v1 = Vector(0.1, 0.2, 0.3)
        v2 = Vector(0.3, 0.6, 0.9)
        result = v1 * 3
        self.assertAlmostEqual(result.x, v2.x)
        self.assertAlmostEqual(result.y, v2.y)
        self.assertAlmostEqual(result.z, v2.z)
    
    def test_large_magnitude(self):
        """Test vector with large magnitude."""
        v = Vector(1e10, 1e10, 1e10)
        mag = v.magnitude()
        self.assertIsInstance(mag, float)
        self.assertGreater(mag, 0)


if __name__ == '__main__':
    unittest.main()
