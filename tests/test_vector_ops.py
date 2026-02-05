"""
Unit tests for matpy.vector.ops module.

Tests all standalone vector operation functions.
"""

import unittest
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matpy.vector.core import Vector
from matpy.vector import ops


class TestBasicOperations(unittest.TestCase):
    """Test basic vector operations."""
    
    def test_dot(self):
        """Test dot product function."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        result = ops.dot(v1, v2)
        self.assertEqual(result, 32)
    
    def test_cross(self):
        """Test cross product function."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        result = ops.cross(v1, v2)
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 1)
    
    def test_magnitude(self):
        """Test magnitude function."""
        v = Vector(3, 4, 0)
        result = ops.magnitude(v)
        self.assertAlmostEqual(result, 5.0)
    
    def test_magnitude_zero_vector(self):
        """Test magnitude of zero vector."""
        v = Vector(0, 0, 0)
        result = ops.magnitude(v)
        self.assertEqual(result, 0.0)
    
    def test_normalize(self):
        """Test normalize function."""
        v = Vector(3, 4, 0)
        result = ops.normalize(v)
        self.assertAlmostEqual(ops.magnitude(result), 1.0)
        self.assertAlmostEqual(result.x, 0.6)
        self.assertAlmostEqual(result.y, 0.8)
    
    def test_normalize_zero_vector(self):
        """Test normalize zero vector."""
        v = Vector(0, 0, 0)
        result = ops.normalize(v)
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)


class TestAngleBetween(unittest.TestCase):
    """Test angle_between function."""
    
    def test_angle_between_perpendicular(self):
        """Test angle between perpendicular vectors."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        angle = ops.angle_between(v1, v2)
        self.assertAlmostEqual(angle, math.pi / 2)
    
    def test_angle_between_parallel(self):
        """Test angle between parallel vectors."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(2, 0, 0)
        angle = ops.angle_between(v1, v2)
        self.assertAlmostEqual(angle, 0.0)
    
    def test_angle_between_opposite(self):
        """Test angle between opposite vectors."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(-1, 0, 0)
        angle = ops.angle_between(v1, v2)
        self.assertAlmostEqual(angle, math.pi)
    
    def test_angle_between_45_degrees(self):
        """Test angle of 45 degrees."""
        v1 = Vector(1, 0, 0)
        v2 = Vector(1, 1, 0)
        angle = ops.angle_between(v1, v2)
        self.assertAlmostEqual(angle, math.pi / 4)
    
    def test_angle_between_zero_vector(self):
        """Test angle with zero vector."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(0, 0, 0)
        angle = ops.angle_between(v1, v2)
        self.assertEqual(angle, 0.0)


class TestProjection(unittest.TestCase):
    """Test projection function."""
    
    def test_projection_basic(self):
        """Test basic projection."""
        v1 = Vector(3, 4, 0)
        v2 = Vector(1, 0, 0)
        result = ops.projection(v1, v2)
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)
    
    def test_projection_perpendicular(self):
        """Test projection of perpendicular vectors."""
        v1 = Vector(0, 1, 0)
        v2 = Vector(1, 0, 0)
        result = ops.projection(v1, v2)
        self.assertAlmostEqual(result.x, 0)
        self.assertAlmostEqual(result.y, 0)
        self.assertAlmostEqual(result.z, 0)
    
    def test_projection_parallel(self):
        """Test projection of parallel vectors."""
        v1 = Vector(3, 0, 0)
        v2 = Vector(2, 0, 0)
        result = ops.projection(v1, v2)
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)
    
    def test_projection_zero_vector(self):
        """Test projection onto zero vector."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(0, 0, 0)
        result = ops.projection(v1, v2)
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)


class TestRejection(unittest.TestCase):
    """Test rejection function."""
    
    def test_rejection_basic(self):
        """Test basic rejection."""
        v1 = Vector(3, 4, 0)
        v2 = Vector(1, 0, 0)
        result = ops.rejection(v1, v2)
        self.assertAlmostEqual(result.x, 0)
        self.assertAlmostEqual(result.y, 4)
        self.assertAlmostEqual(result.z, 0)
    
    def test_rejection_perpendicular(self):
        """Test rejection of perpendicular vectors."""
        v1 = Vector(0, 1, 0)
        v2 = Vector(1, 0, 0)
        result = ops.rejection(v1, v2)
        self.assertAlmostEqual(result.x, 0)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.z, 0)
    
    def test_projection_plus_rejection(self):
        """Test that projection + rejection = original vector."""
        v1 = Vector(5, 7, 3)
        v2 = Vector(2, 1, 4)
        proj = ops.projection(v1, v2)
        rej = ops.rejection(v1, v2)
        result = proj + rej
        self.assertAlmostEqual(result.x, v1.x)
        self.assertAlmostEqual(result.y, v1.y)
        self.assertAlmostEqual(result.z, v1.z)


class TestReflection(unittest.TestCase):
    """Test reflect function."""
    
    def test_reflect_horizontal(self):
        """Test reflection across horizontal surface."""
        v = Vector(1, -1, 0)
        normal = Vector(0, 1, 0)
        result = ops.reflect(v, normal)
        self.assertAlmostEqual(result.x, 1)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.z, 0)
    
    def test_reflect_vertical(self):
        """Test reflection across vertical surface."""
        v = Vector(-1, 1, 0)
        normal = Vector(1, 0, 0)
        result = ops.reflect(v, normal)
        self.assertAlmostEqual(result.x, 1)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.z, 0)
    
    def test_reflect_perpendicular(self):
        """Test reflection perpendicular to surface."""
        v = Vector(0, -1, 0)
        normal = Vector(0, 1, 0)
        result = ops.reflect(v, normal)
        self.assertAlmostEqual(result.x, 0)
        self.assertAlmostEqual(result.y, 1)
        self.assertAlmostEqual(result.z, 0)
    
    def test_reflect_magnitude_preserved(self):
        """Test that reflection preserves magnitude."""
        v = Vector(3, -4, 2)
        normal = Vector(0, 1, 0)
        result = ops.reflect(v, normal)
        self.assertAlmostEqual(ops.magnitude(v), ops.magnitude(result))


class TestLerp(unittest.TestCase):
    """Test linear interpolation function."""
    
    def test_lerp_start(self):
        """Test lerp at t=0 returns start vector."""
        v1 = Vector(0, 0, 0)
        v2 = Vector(10, 10, 10)
        result = ops.lerp(v1, v2, 0.0)
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)
    
    def test_lerp_end(self):
        """Test lerp at t=1 returns end vector."""
        v1 = Vector(0, 0, 0)
        v2 = Vector(10, 10, 10)
        result = ops.lerp(v1, v2, 1.0)
        self.assertEqual(result.x, 10)
        self.assertEqual(result.y, 10)
        self.assertEqual(result.z, 10)
    
    def test_lerp_midpoint(self):
        """Test lerp at t=0.5 returns midpoint."""
        v1 = Vector(0, 0, 0)
        v2 = Vector(10, 20, 30)
        result = ops.lerp(v1, v2, 0.5)
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 10)
        self.assertEqual(result.z, 15)
    
    def test_lerp_quarter(self):
        """Test lerp at t=0.25."""
        v1 = Vector(0, 0, 0)
        v2 = Vector(100, 100, 100)
        result = ops.lerp(v1, v2, 0.25)
        self.assertEqual(result.x, 25)
        self.assertEqual(result.y, 25)
        self.assertEqual(result.z, 25)
    
    def test_lerp_extrapolation(self):
        """Test lerp with t > 1 (extrapolation)."""
        v1 = Vector(0, 0, 0)
        v2 = Vector(10, 10, 10)
        result = ops.lerp(v1, v2, 2.0)
        self.assertEqual(result.x, 20)
        self.assertEqual(result.y, 20)
        self.assertEqual(result.z, 20)
    
    def test_lerp_negative_t(self):
        """Test lerp with t < 0 (extrapolation backwards)."""
        v1 = Vector(10, 10, 10)
        v2 = Vector(20, 20, 20)
        result = ops.lerp(v1, v2, -1.0)
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)


class TestOperationsIntegration(unittest.TestCase):
    """Integration tests combining multiple operations."""
    
    def test_orthogonal_decomposition(self):
        """Test decomposing a vector into parallel and perpendicular components."""
        v = Vector(5, 5, 0)
        basis = Vector(1, 0, 0)
        
        parallel = ops.projection(v, basis)
        perpendicular = ops.rejection(v, basis)
        
        # Parallel component should be along basis
        self.assertAlmostEqual(parallel.y, 0)
        
        # Perpendicular should be orthogonal
        self.assertAlmostEqual(ops.dot(perpendicular, basis), 0)
        
        # Sum should equal original
        result = parallel + perpendicular
        self.assertAlmostEqual(result.x, v.x)
        self.assertAlmostEqual(result.y, v.y)
    
    def test_normalize_then_magnitude(self):
        """Test that normalizing a vector gives unit length."""
        v = Vector(3, 4, 12)
        normalized = ops.normalize(v)
        mag = ops.magnitude(normalized)
        self.assertAlmostEqual(mag, 1.0)
    
    def test_cross_product_orthogonality(self):
        """Test that cross product is orthogonal to both inputs."""
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        cross = ops.cross(v1, v2)
        
        self.assertAlmostEqual(ops.dot(cross, v1), 0)
        self.assertAlmostEqual(ops.dot(cross, v2), 0)


if __name__ == '__main__':
    unittest.main()
