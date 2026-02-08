"""
Test suite for eigenvalue and eigenvector computations.

Tests complex eigenvalue support for 2x2 and 3x3 matrices.
"""

import unittest
import math
from matpy.matrix.core import Matrix
from matpy.error import NotSquareError, NotImplementedError


class TestEigenvalues2x2(unittest.TestCase):
    """Test eigenvalue computation for 2x2 matrices."""
    
    def test_real_eigenvalues(self):
        """Test 2x2 matrix with real eigenvalues."""
        # Matrix [[4, 1], [2, 3]] has eigenvalues 5 and 2
        m = Matrix(2, 2, [[4, 1], [2, 3]])
        eigenvals = Matrix.eigenvalues(m)
        
        # Sort for consistent comparison
        eigenvals_sorted = sorted([float(ev.real) if isinstance(ev, complex) else float(ev) 
                                   for ev in eigenvals], reverse=True)
        self.assertAlmostEqual(eigenvals_sorted[0], 5.0, places=10)
        self.assertAlmostEqual(eigenvals_sorted[1], 2.0, places=10)
    
    def test_complex_eigenvalues_rotation(self):
        """Test 2x2 rotation matrix with complex eigenvalues."""
        # Rotation matrix [[0, -1], [1, 0]] has eigenvalues ±i
        m = Matrix(2, 2, [[0, -1], [1, 0]])
        eigenvals = Matrix.eigenvalues(m)
        
        self.assertEqual(len(eigenvals), 2)
        # Should be complex conjugate pairs
        self.assertIsInstance(eigenvals[0], complex)
        self.assertIsInstance(eigenvals[1], complex)
        
        # Check they are ±i
        self.assertAlmostEqual(eigenvals[0].real, 0.0, places=10)
        self.assertAlmostEqual(eigenvals[1].real, 0.0, places=10)
        self.assertAlmostEqual(abs(eigenvals[0].imag), 1.0, places=10)
        self.assertAlmostEqual(abs(eigenvals[1].imag), 1.0, places=10)
        
        # Verify conjugate pair
        self.assertAlmostEqual(eigenvals[0].imag, -eigenvals[1].imag, places=10)
    
    def test_complex_eigenvalues_general(self):
        """Test 2x2 matrix with general complex eigenvalues."""
        # Matrix [[1, -2], [1, -1]] has eigenvalues i and -i
        m = Matrix(2, 2, [[1, -2], [1, -1]])
        eigenvals = Matrix.eigenvalues(m)
        
        self.assertEqual(len(eigenvals), 2)
        self.assertIsInstance(eigenvals[0], complex)
        self.assertIsInstance(eigenvals[1], complex)
        
        # Both should have real part 0 (trace/2 = 0/2 = 0)
        self.assertAlmostEqual(eigenvals[0].real, 0.0, places=10)
        self.assertAlmostEqual(eigenvals[1].real, 0.0, places=10)
    
    def test_repeated_eigenvalues(self):
        """Test 2x2 matrix with repeated eigenvalues."""
        # Matrix [[3, 0], [0, 3]] has repeated eigenvalue 3
        m = Matrix(2, 2, [[3, 0], [0, 3]])
        eigenvals = Matrix.eigenvalues(m)
        
        self.assertEqual(len(eigenvals), 2)
        self.assertAlmostEqual(float(eigenvals[0]), 3.0, places=10)
        self.assertAlmostEqual(float(eigenvals[1]), 3.0, places=10)
    
    def test_1x1_matrix(self):
        """Test 1x1 matrix eigenvalue."""
        m = Matrix(1, 1, [[5]])
        eigenvals = Matrix.eigenvalues(m)
        
        self.assertEqual(len(eigenvals), 1)
        self.assertEqual(eigenvals[0], 5.0)


class TestEigenvalues3x3(unittest.TestCase):
    """Test eigenvalue computation for 3x3 matrices."""
    
    def test_real_eigenvalues_diagonal(self):
        """Test 3x3 diagonal matrix with real eigenvalues."""
        m = Matrix(3, 3, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        eigenvals = Matrix.eigenvalues(m)
        
        self.assertEqual(len(eigenvals), 3)
        eigenvals_sorted = sorted([float(ev.real) if isinstance(ev, complex) else float(ev) 
                                   for ev in eigenvals], reverse=True)
        self.assertAlmostEqual(eigenvals_sorted[0], 3.0, places=10)
        self.assertAlmostEqual(eigenvals_sorted[1], 2.0, places=10)
        self.assertAlmostEqual(eigenvals_sorted[2], 1.0, places=10)
    
    def test_real_eigenvalues_general(self):
        """Test 3x3 general matrix with real eigenvalues."""
        # Matrix with known eigenvalues
        m = Matrix(3, 3, [[6, -1, 0], [-1, 5, -1], [0, -1, 4]])
        eigenvals = Matrix.eigenvalues(m)
        
        self.assertEqual(len(eigenvals), 3)
        # All should be real
        for ev in eigenvals:
            if isinstance(ev, complex):
                self.assertAlmostEqual(ev.imag, 0.0, places=10)
    
    def test_complex_eigenvalues_3x3(self):
        """Test 3x3 matrix with complex eigenvalues."""
        # Matrix with one real and two complex conjugate eigenvalues
        m = Matrix(3, 3, [[0, -1, 0], [1, 0, 0], [0, 0, 2]])
        eigenvals = Matrix.eigenvalues(m)
        
        self.assertEqual(len(eigenvals), 3)
        
        # Separate real and complex eigenvalues
        real_evals = [ev for ev in eigenvals if not isinstance(ev, complex) or abs(ev.imag) < 1e-10]
        complex_evals = [ev for ev in eigenvals if isinstance(ev, complex) and abs(ev.imag) >= 1e-10]
        
        # Should have 1 real and 2 complex conjugates
        self.assertEqual(len(real_evals), 1)
        self.assertEqual(len(complex_evals), 2)
        
        # Real eigenvalue should be 2
        self.assertAlmostEqual(float(real_evals[0]), 2.0, places=10)
        
        # Complex eigenvalues should be conjugate pairs (±i)
        self.assertAlmostEqual(complex_evals[0].real, complex_evals[1].real, places=10)
        self.assertAlmostEqual(complex_evals[0].imag, -complex_evals[1].imag, places=10)


class TestEigenvectors2x2(unittest.TestCase):
    """Test eigenvector computation for 2x2 matrices."""
    
    def test_real_eigenvectors(self):
        """Test 2x2 matrix with real eigenvectors."""
        m = Matrix(2, 2, [[4, 1], [2, 3]])
        eigenvecs = Matrix.eigenvectors(m)
        
        self.assertEqual(len(eigenvecs), 2)
        # Each eigenvector should be normalized
        for vec in eigenvecs:
            magnitude_sq = sum(v**2 if not isinstance(v, complex) else abs(v)**2 for v in vec)
            self.assertAlmostEqual(magnitude_sq, 1.0, places=10)
    
    def test_complex_eigenvectors(self):
        """Test 2x2 matrix with complex eigenvectors."""
        # Rotation matrix
        m = Matrix(2, 2, [[0, -1], [1, 0]])
        eigenvecs = Matrix.eigenvectors(m)
        
        self.assertEqual(len(eigenvecs), 2)
        # Eigenvectors should be normalized (using complex magnitude)
        for vec in eigenvecs:
            # For complex vectors, magnitude = sqrt(sum of |vi|^2)
            magnitude_sq = sum(abs(v)**2 for v in vec)
            self.assertAlmostEqual(magnitude_sq, 1.0, places=10)
    
    def test_diagonal_eigenvectors(self):
        """Test diagonal matrix eigenvectors."""
        m = Matrix(2, 2, [[3, 0], [0, 5]])
        eigenvecs = Matrix.eigenvectors(m)
        
        self.assertEqual(len(eigenvecs), 2)
        # Diagonal matrix should have standard basis vectors (approximately)
        # Each should be normalized
        for vec in eigenvecs:
            magnitude_sq = sum(v**2 if not isinstance(v, complex) else abs(v)**2 for v in vec)
            self.assertAlmostEqual(magnitude_sq, 1.0, places=10)


class TestEigenvectors3x3(unittest.TestCase):
    """Test eigenvector computation for 3x3 matrices."""
    
    def test_diagonal_eigenvectors(self):
        """Test 3x3 diagonal matrix eigenvectors."""
        m = Matrix(3, 3, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        eigenvecs = Matrix.eigenvectors(m)
        
        self.assertEqual(len(eigenvecs), 3)
        # Each eigenvector should be normalized
        for vec in eigenvecs:
            magnitude_sq = sum(v**2 if not isinstance(v, complex) else abs(v)**2 for v in vec)
            self.assertAlmostEqual(magnitude_sq, 1.0, places=10)
    
    def test_3x3_with_complex_eigenvectors(self):
        """Test 3x3 matrix with complex eigenvectors."""
        m = Matrix(3, 3, [[0, -1, 0], [1, 0, 0], [0, 0, 2]])
        eigenvecs = Matrix.eigenvectors(m)
        
        self.assertEqual(len(eigenvecs), 3)
        # Each should be normalized (complex or real)
        for vec in eigenvecs:
            magnitude_sq = sum(abs(v)**2 for v in vec)
            self.assertAlmostEqual(magnitude_sq, 1.0, places=10)


class TestEigenvalueErrors(unittest.TestCase):
    """Test error handling for eigenvalue computations."""
    
    def test_non_square_matrix(self):
        """Test that non-square matrices raise error."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(NotSquareError):
            Matrix.eigenvalues(m)
    
    def test_large_matrix_not_implemented(self):
        """Test that 4x4+ matrices raise NotImplementedError."""
        m = Matrix(4, 4, [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
        with self.assertRaises(NotImplementedError):
            Matrix.eigenvalues(m)
    
    def test_eigenvectors_non_square(self):
        """Test that eigenvectors for non-square matrices raise error."""
        m = Matrix(2, 3, [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(NotSquareError):
            Matrix.eigenvectors(m)
    
    def test_eigenvectors_large_matrix(self):
        """Test that eigenvectors for 4x4+ matrices raise NotImplementedError."""
        m = Matrix(4, 4, [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
        with self.assertRaises(NotImplementedError):
            Matrix.eigenvectors(m)


class TestEigenvalueProperties(unittest.TestCase):
    """Test mathematical properties of eigenvalues."""
    
    def test_trace_equals_sum_of_eigenvalues(self):
        """Test that trace equals sum of eigenvalues."""
        m = Matrix(2, 2, [[4, 1], [2, 3]])
        eigenvals = Matrix.eigenvalues(m)
        trace = m.trace()
        
        eigenval_sum = sum(ev.real if isinstance(ev, complex) else ev for ev in eigenvals)
        self.assertAlmostEqual(trace, eigenval_sum, places=10)
    
    def test_determinant_equals_product_of_eigenvalues(self):
        """Test that determinant equals product of eigenvalues."""
        m = Matrix(2, 2, [[4, 1], [2, 3]])
        eigenvals = Matrix.eigenvalues(m)
        det = m.determinant()
        
        eigenval_product = 1
        for ev in eigenvals:
            if isinstance(ev, complex):
                eigenval_product *= abs(ev)  # For complex, use magnitude
            else:
                eigenval_product *= ev
        
        # For real eigenvalues, this should match exactly
        if all(not isinstance(ev, complex) or abs(ev.imag) < 1e-10 for ev in eigenvals):
            self.assertAlmostEqual(det, eigenval_product, places=10)
    
    def test_complex_eigenvalues_are_conjugate_pairs(self):
        """Test that complex eigenvalues come in conjugate pairs."""
        m = Matrix(2, 2, [[0, -1], [1, 0]])
        eigenvals = Matrix.eigenvalues(m)
        
        # For real matrices, complex eigenvalues come in conjugate pairs
        if len(eigenvals) == 2 and all(isinstance(ev, complex) for ev in eigenvals):
            ev1, ev2 = eigenvals
            self.assertAlmostEqual(ev1.real, ev2.real, places=10)
            self.assertAlmostEqual(ev1.imag, -ev2.imag, places=10)


if __name__ == '__main__':
    unittest.main()
