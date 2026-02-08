"""
Matrix class implementation.

Provides a comprehensive Matrix class with arithmetic operations,
matrix-specific operations, and Python dunder methods.
"""

from __future__ import annotations
from typing import List, Union

from ..error import (
    ValidationError,
    ShapeError,
    NotSquareError,
    SingularMatrixError,
    IndexError as MatPyIndexError,
    NotImplementedError as MatPyNotImplementedError,
    require_square,
    require_compatible_for_multiplication,
)
from ..core import validate, utils

# Default tolerance for floating point comparisons
DEFAULT_TOLERANCE = 1e-10


class Matrix:
    """
    A matrix class supporting common matrix operations and Python protocols.
    
    Attributes:
        rows (int): Number of rows in the matrix
        cols (int): Number of columns in the matrix
        data (List[List[float]]): 2D list containing matrix elements
    """
    
    __slots__ = ('rows', 'cols', 'data')
    
    # ==================== Initialization ====================
    
    def __init__(self, rows: int, cols: int, data: List[List[float]] = None) -> None:
        """
        Initialize a matrix.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            data: Optional 2D list of values. If None, creates zero matrix.
        
        Raises:
            ValueError: If data dimensions don't match rows and cols
        """
        validate.validate_matrix_dimensions(rows, cols)
        self.rows = rows
        self.cols = cols
        
        if data is None:
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]
        else:
            validate.validate_data_shape(data, rows, cols, "matrix data")
            self.data = data
    
    # ==================== String Representation ====================
    
    def __str__(self) -> str:
        """Return readable string representation with tab-separated values."""
        return utils.format_matrix_string(self.data)
    
    def __repr__(self) -> str:
        """Return official string representation."""
        return f"Matrix({self.rows}x{self.cols}, data={self.data})"
    
    def __format__(self, format_spec: str) -> str:
        """Return formatted string representation."""
        formatted_rows = []
        for row in self.data:
            formatted_row = '\t'.join(format(value, format_spec) for value in row)
            formatted_rows.append(formatted_row)
        return '\n'.join(formatted_rows)
    
    # ==================== Arithmetic Operators ====================
    
    def __add__(self, other: Matrix) -> Matrix:
        """Add two matrices element-wise."""
        validate.validate_same_shape(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "addition"
        )
        
        result_data = [
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(self.rows, self.cols, result_data)
    
    def __sub__(self, other: Matrix) -> Matrix:
        """Subtract two matrices element-wise."""
        validate.validate_same_shape(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "subtraction"
        )
        
        result_data = [
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(self.rows, self.cols, result_data)
    
    def _scalar_multiply(self, scalar: float) -> Matrix:
        """Helper: multiply matrix by scalar."""
        result_data = [
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(self.rows, self.cols, result_data)
    
    def _matrix_multiply(self, other: Matrix) -> Matrix:
        """Helper: multiply two matrices."""
        require_compatible_for_multiplication(self, other)
        result_data = [
            [sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) for j in range(other.cols)]
            for i in range(self.rows)
        ]
        return Matrix(self.rows, other.cols, result_data)
    
    def __mul__(self, other: Union[Matrix, float]) -> Matrix:
        """
        Matrix multiplication or scalar multiplication.
        
        Args:
            other: Another Matrix or a scalar value
        
        Returns:
            Result of matrix multiplication or scalar multiplication
        """
        if isinstance(other, (int, float)):
            return self._scalar_multiply(other)
        return self._matrix_multiply(other)
    
    def __rmul__(self, scalar: float) -> Matrix:
        """Reflected multiplication: scalar * matrix."""
        return self._scalar_multiply(scalar)
    
    def __truediv__(self, scalar: float) -> Matrix:
        """Divide matrix by scalar."""
        return self._scalar_multiply(1.0 / scalar)
    
    def __matmul__(self, other: Matrix) -> Matrix:
        """Matrix multiplication using @ operator (Python 3.5+)."""
        return self._matrix_multiply(other)
    
    def __pow__(self, exponent: int) -> Matrix:
        """
        Matrix exponentiation (repeated matrix multiplication).
        
        Args:
            exponent: Integer exponent (must be non-negative)
        
        Returns:
            Matrix raised to the power
        
        Raises:
            ValueError: If matrix is not square or exponent is negative
        """
        require_square(self, "Matrix exponentiation")
        
        if exponent < 0:
            raise ValidationError("Negative exponents not supported. Use inverse() instead.")
        
        if exponent == 0:
            identity_data = [[1 if i == j else 0 for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(self.rows, self.cols, identity_data)
        
        if exponent == 1:
            return self.__copy__()
        
        # Binary exponentiation for O(log n) efficiency
        result = None
        base = self.__copy__()
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = base if result is None else result @ base
            exponent //= 2
            if exponent > 0:
                base = base @ base
        
        return result
    
    # ==================== Reflected Arithmetic Operators ====================
    
    def __radd__(self, other: Matrix) -> Matrix:
        """Reflected addition: other + self."""
        return self.__add__(other)
    
    def __rsub__(self, other: Matrix) -> Matrix:
        """Reflected subtraction: other - self."""
        validate.validate_same_shape(
            (other.rows, other.cols),
            (self.rows, self.cols),
            "subtraction"
        )
        
        result_data = [
            [other.data[i][j] - self.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(self.rows, self.cols, result_data)
    
    # ==================== In-Place Operators ====================
    
    def __iadd__(self, other: Matrix) -> Matrix:
        """In-place addition: self += other."""
        validate.validate_same_shape(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "addition"
        )
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] += other.data[i][j]
        return self
    
    def __isub__(self, other: Matrix) -> Matrix:
        """In-place subtraction: self -= other."""
        validate.validate_same_shape(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "subtraction"
        )
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] -= other.data[i][j]
        return self
    
    def __imul__(self, other: Union[Matrix, float]) -> Matrix:
        """In-place multiplication: self *= other."""
        # Scalar multiplication
        if isinstance(other, (int, float)):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= other
            return self
        
        # Matrix multiplication - cannot be done in-place, return new matrix
        return self.__mul__(other)
    
    def __itruediv__(self, scalar: float) -> Matrix:
        """In-place division: self /= scalar."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] /= scalar
        return self
    
    def __imatmul__(self, other: Matrix) -> Matrix:
        """In-place matrix multiplication: self @= other."""
        # Matrix multiplication cannot be truly in-place, return new result
        return self.__matmul__(other)
    
    # ==================== Unary Operators ====================
    
    def __neg__(self) -> Matrix:
        """Negate all elements in the matrix."""
        result_data = [[-value for value in row] for row in self.data]
        return Matrix(self.rows, self.cols, result_data)
    
    def __pos__(self) -> Matrix:
        """Unary plus (returns copy)."""
        return self.__copy__()
    
    def __abs__(self) -> float:
        """
        Return the Frobenius norm of the matrix.
        
        The Frobenius norm is the square root of the sum of squares of all elements.
        
        Returns:
            The Frobenius norm
        """
        total = sum(value ** 2 for row in self.data for value in row)
        return total ** 0.5
    
    def __round__(self, ndigits: int = 0) -> Matrix:
        """
        Round all matrix elements to n digits.
        
        Args:
            ndigits: Number of decimal places (default: 0)
        
        Returns:
            New matrix with rounded elements
        """
        result_data = [
            [round(value, ndigits) for value in row]
            for row in self.data
        ]
        return Matrix(self.rows, self.cols, result_data)
    
    # ==================== Comparison Operators ====================
    
    def __eq__(self, other: object) -> bool:
        """Check if two matrices are equal."""
        if not isinstance(other, Matrix):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        return all(
            self.data[i][j] == other.data[i][j] 
            for i in range(self.rows) 
            for j in range(self.cols)
        )
    
    def __ne__(self, other: object) -> bool:
        """Check if two matrices are not equal."""
        return not self.__eq__(other)
    
    def __lt__(self, other: Matrix) -> bool:
        """Less than comparison based on Frobenius norm."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return abs(self) < abs(other)
    
    def __le__(self, other: Matrix) -> bool:
        """Less than or equal comparison based on Frobenius norm."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return abs(self) <= abs(other)
    
    def __gt__(self, other: Matrix) -> bool:
        """Greater than comparison based on Frobenius norm."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return abs(self) > abs(other)
    
    def __ge__(self, other: Matrix) -> bool:
        """Greater than or equal comparison based on Frobenius norm."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return abs(self) >= abs(other)
    
    # ==================== Type Conversion ====================
    
    def __bool__(self) -> bool:
        """Return True if matrix has any non-zero elements."""
        return any(any(value != 0 for value in row) for row in self.data)
    
    def __hash__(self) -> int:
        """Return hash of matrix (makes it hashable)."""
        return hash(tuple(tuple(row) for row in self.data))
    
    # ==================== Container/Sequence Methods ====================
    
    def __getitem__(self, index: Union[int, tuple]) -> Union[List[float], float]:
        """
        Get matrix element(s) by index.
        
        Args:
            index: Either an integer (for row access) or tuple (row, col) for element access
            
        Returns:
            For integer index: Returns a row as a list
            For tuple index: Returns the element at [row, col]
        """
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValidationError(f"Matrix indexing requires exactly 2 indices, got {len(index)}")
            row, col = index
            validate.validate_index(row, self.rows, "row index")
            validate.validate_index(col, self.cols, "column index")
            return self.data[row][col]
        else:
            validate.validate_index(index, self.rows, "row index")
            return self.data[index]
    
    def __setitem__(self, index: Union[int, tuple], value: Union[List[float], float]) -> None:
        """
        Set matrix element(s) by index.
        
        Args:
            index: Either an integer (for row assignment) or tuple (row, col) for element assignment
            value: Either a list (for row assignment) or float (for element assignment)
        """
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValidationError(f"Matrix indexing requires exactly 2 indices, got {len(index)}")
            row, col = index
            validate.validate_index(row, self.rows, "row index")
            validate.validate_index(col, self.cols, "column index")
            self.data[row][col] = value
        else:
            validate.validate_index(index, self.rows, "row index")
            validate.validate_dimensions_match(len(value), self.cols, "row assignment")
            self.data[index] = value
    
    def __iter__(self):
        """Iterate over rows of the matrix."""
        return iter(self.data)
    
    def __contains__(self, item: float) -> bool:
        """Check if value exists in matrix."""
        return any(item in row for row in self.data)
    
    def __len__(self) -> int:
        """Return total number of elements."""
        return self.rows * self.cols
    
    # ==================== Copying ====================
    
    def __copy__(self) -> Matrix:
        """Create a shallow copy of the matrix."""
        new_data = utils.deep_copy_2d(self.data)
        return Matrix(self.rows, self.cols, new_data)
    
    def __deepcopy__(self, memo) -> Matrix:
        """Create a deep copy of the matrix."""
        new_data = utils.deep_copy_2d(self.data)
        return Matrix(self.rows, self.cols, new_data)
    
    # ==================== Matrix Operations ====================
    
    def transpose(self) -> Matrix:
        """Return the transpose of the matrix."""
        result_data = utils.transpose_list(self.data)
        return Matrix(self.cols, self.rows, result_data)
    
    def trace(self) -> float:
        """
        Calculate the trace (sum of diagonal elements).
        
        Returns:
            The trace of the matrix
        
        Raises:
            ValueError: If matrix is not square
        """
        require_square(self, "Trace")
        return sum(self.data[i][i] for i in range(self.rows))
    
    def is_symmetric(self) -> bool:
        """Check if matrix is symmetric (M = M^T)."""
        if self.rows != self.cols:
            return False
        return all(
            utils.is_close(self.data[i][j], self.data[j][i]) 
            for i in range(self.rows) 
            for j in range(self.cols)
        )
    
    def is_square(self) -> bool:
        """Check if matrix is square."""
        return self.rows == self.cols
    
    # ==================== Advanced Matrix Operations ====================
    
    def _minor(self, row: int, col: int) -> Matrix:
        """
        Get the minor matrix by removing specified row and column.
        
        Args:
            row: Row index to remove
            col: Column index to remove
        
        Returns:
            Matrix with specified row and column removed
        """
        minor_data = [
            [self.data[i][j] for j in range(self.cols) if j != col]
            for i in range(self.rows) if i != row
        ]
        return Matrix(self.rows - 1, self.cols - 1, minor_data)
    
    def cofactor(self, row: int, col: int) -> float:
        """
        Calculate the cofactor at position (row, col).
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
        
        Returns:
            The cofactor value
        
        Raises:
            ValueError: If matrix is not square or indices are invalid
        """
        require_square(self, "Cofactor")
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise MatPyIndexError((row, col), f"0-{self.rows-1}, 0-{self.cols-1}")
        
        minor = self._minor(row, col)
        sign = (-1) ** (row + col)
        return sign * minor.determinant()
    
    def determinant(self) -> float:
        """
        Calculate the determinant of the matrix using cofactor expansion.
        
        Returns:
            The determinant value
        
        Raises:
            ValueError: If matrix is not square
        """
        require_square(self, "Determinant")
        
        # Base case: 1x1 matrix
        if self.rows == 1:
            return self.data[0][0]
        
        # Base case: 2x2 matrix
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        # Recursive case: use cofactor expansion along first row
        det = 0
        for col in range(self.cols):
            det += self.data[0][col] * self.cofactor(0, col)
        
        return det
    
    def adjugate(self) -> Matrix:
        """
        Calculate the adjugate (adjoint) matrix.
        
        Returns:
            The adjugate matrix (transpose of cofactor matrix)
        
        Raises:
            ValueError: If matrix is not square
        """
        require_square(self, "Adjugate")
        
        # Special case: 1x1 matrix
        if self.rows == 1:
            return Matrix(1, 1, [[1]])
        
        # Calculate cofactor matrix
        cofactor_data = [
            [self.cofactor(i, j) for j in range(self.cols)]
            for i in range(self.rows)
        ]
        cofactor_matrix = Matrix(self.rows, self.cols, cofactor_data)
        
        # Return transpose of cofactor matrix
        return cofactor_matrix.transpose()
    
    def inverse(self) -> Matrix:
        """
        Calculate the inverse of the matrix using the adjugate method.
        
        Returns:
            The inverse matrix
        
        Raises:
            ValueError: If matrix is not square or is singular
        """
        require_square(self, "Matrix inverse")
        
        det = self.determinant()
        if validate.approx_zero(det):  # Near-zero determinant
            raise SingularMatrixError()
        
        # For 1x1 matrix
        if self.rows == 1:
            return Matrix(1, 1, [[1 / det]])
        
        # Use adjugate method: A^(-1) = adj(A) / det(A)
        adj = self.adjugate()
        return adj / det
    
    def rank(self) -> int:
        """
        Calculate the rank of the matrix using row reduction.
        
        Returns:
            The rank (number of linearly independent rows)
        """
        # Create a copy to avoid modifying original
        temp = self.__copy__()
        rank = 0
        
        for col in range(min(temp.rows, temp.cols)):
            # Find pivot
            pivot_row = None
            for row in range(rank, temp.rows):
                if not validate.approx_zero(temp.data[row][col]):
                    pivot_row = row
                    break
            
            if pivot_row is None:
                continue
            
            # Swap rows if necessary
            if pivot_row != rank:
                temp.data[rank], temp.data[pivot_row] = temp.data[pivot_row], temp.data[rank]
            
            # Scale pivot row
            pivot = temp.data[rank][col]
            for j in range(temp.cols):
                temp.data[rank][j] /= pivot
            
            # Eliminate column in other rows
            for row in range(temp.rows):
                if row != rank:
                    factor = temp.data[row][col]
                    for j in range(temp.cols):
                        temp.data[row][j] -= factor * temp.data[rank][j]
            
            rank += 1
        
        return rank
    
    def is_singular(self) -> bool:
        """
        Check if matrix is singular (determinant = 0).
        
        Returns:
            True if matrix is singular, False otherwise
        
        Raises:
            ValueError: If matrix is not square
        """
        require_square(self, "Singularity check")
        
        return validate.approx_zero(self.determinant())
    
    def is_invertible(self) -> bool:
        """
        Check if matrix is invertible (non-singular).
        
        Returns:
            True if matrix is invertible, False otherwise
        
        Raises:
            ValueError: If matrix is not square
        """
        if not self.is_square():
            return False
        
        return not self.is_singular()
    
    def is_orthogonal(self) -> bool:
        """
        Check if matrix is orthogonal (M^T * M = I).
        
        Returns:
            True if matrix is orthogonal, False otherwise
        
        Raises:
            ValueError: If matrix is not square
        """
        if not self.is_square():
            return False
        
        # Calculate M^T * M
        product = self.transpose() * self
        
        # Check if it equals identity matrix
        for i in range(self.rows):
            for j in range(self.cols):
                expected = 1.0 if i == j else 0.0
                if not utils.is_close(product.data[i][j], expected):
                    return False
        
        return True
    
    def is_diagonal(self) -> bool:
        """
        Check if matrix is diagonal (all non-diagonal elements are zero).
        
        Returns:
            True if diagonal, False otherwise
        """
        if not self.is_square():
            return False
        
        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and not validate.approx_zero(self.data[i][j]):
                    return False
        return True
    
    def is_identity(self) -> bool:
        """
        Check if matrix is an identity matrix.
        
        Returns:
            True if identity matrix, False otherwise
        """
        if not self.is_square():
            return False
        
        for i in range(self.rows):
            for j in range(self.cols):
                expected = 1.0 if i == j else 0.0
                if not utils.is_close(self.data[i][j], expected):
                    return False
        return True
    
    def is_upper_triangular(self) -> bool:
        """
        Check if matrix is upper triangular (all elements below diagonal are zero).
        
        Returns:
            True if upper triangular, False otherwise
        """
        if not self.is_square():
            return False
        
        for i in range(self.rows):
            for j in range(i):
                if not validate.approx_zero(self.data[i][j]):
                    return False
        return True
    
    def is_lower_triangular(self) -> bool:
        """
        Check if matrix is lower triangular (all elements above diagonal are zero).
        
        Returns:
            True if lower triangular, False otherwise
        """
        if not self.is_square():
            return False
        
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if not validate.approx_zero(self.data[i][j]):
                    return False
        return True
    
    # ==================== Complex Operations (Simplified Implementations) ====================
    
    @staticmethod
    def eigenvalues(A: Matrix) -> List[Union[float, complex]]:
        """
        Calculate eigenvalues of the matrix.
        
        Note: This is a simplified implementation using characteristic polynomial
        for small matrices. For larger matrices, use numerical libraries like NumPy.
        
        Args:
            matrix: The matrix to calculate eigenvalues for
        
        Returns:
            List of eigenvalues (real or complex)
        
        Raises:
            ValueError: If matrix is not square
            NotImplementedError: For matrices larger than 3x3
        
        Example:
            >>> A = Matrix(2, 2, [[4, 2], [1, 3]])
            >>> eigenvals = Matrix.eigenvalues(A)
            >>> # eigenvals ≈ [5.0, 2.0]
        """
        require_square(A, "Eigenvalues")
        
        # For 1x1 matrix
        if A.rows == 1: return [A.data[0][0]]
        
        # For 2x2 matrix: solve characteristic equation λ² - tr(A)λ + det(A) = 0
        if A.rows == 2:
            disc = A.trace() ** 2 - 4 * A.determinant()
            
            if disc < 0:
                # Complex eigenvalues: λ = (tr ± i√|disc|) / 2
                Re = A.trace() / 2
                Im = (abs(disc) ** 0.5) / 2
                λ1 = complex(Re, Im)
                λ2 = complex(Re, -Im)
                return [λ1, λ2]
            
            sqrt_disc = disc ** 0.5
            λ1 = (A.trace() + sqrt_disc) / 2
            λ2 = (A.trace() - sqrt_disc) / 2
            return [λ1, λ2]
        
        # For 3x3 matrix: use characteristic polynomial
        if A.rows == 3:
            # Characteristic polynomial: det(A - λI) = 0
            # This gives: -λ³ + c₂λ² + c₁λ + c₀ = 0
            # where c₂ = tr(A), c₁ = sum of principal minors, c₀ = -det(A)
            
            import math
            
            # Sum of 2x2 principal minors
            minor_sum = (
                A.data[0][0] * A.data[1][1] - A.data[0][1] * A.data[1][0] +
                A.data[0][0] * A.data[2][2] - A.data[0][2] * A.data[2][0] +
                A.data[1][1] * A.data[2][2] - A.data[1][2] * A.data[2][1]
            )
            
            # Solve cubic: λ³ - tr·λ² + minor_sum·λ - det = 0
            # Use Cardano's method
            a = -A.trace()
            b = minor_sum
            c = -A.determinant()
            
            # Convert to depressed cubic: t³ + pt + q = 0 where λ = t - a/3
            p = b - a ** 2 / 3
            q = 2 * a ** 3 / 27 - a * b / 3 + c
            
            # Calculate discriminant
            disc = -(4 * p * p * p + 27 * q * q)
            
            if disc > 0:
                # Three distinct real roots
                m = 2 * math.sqrt(-p / 3)
                theta = math.acos(3 * q / (p * m)) / 3
                
                λ1 = m * math.cos(theta) - a / 3
                λ2 = m * math.cos(theta - 2 * math.pi / 3) - a / 3
                λ3 = m * math.cos(theta - 4 * math.pi / 3) - a / 3
                
                return sorted([λ1, λ2, λ3], reverse=True)
            elif abs(disc) < DEFAULT_TOLERANCE:
                # Repeated roots
                if abs(p) < DEFAULT_TOLERANCE:
                    # Triple root
                    return [-a / 3, -a / 3, -a / 3]
                else:
                    # One simple and one double root
                    λ1 = 3 * q / p - a / 3
                    λ2 = -3 * q / (2 * p) - a / 3
                    return sorted([λ1, λ2, λ2], reverse=True)
            else:
                # One real root and two complex conjugates
                if p == 0:
                    λ1 = -q ** (1/3) - a / 3
                else: 
                    # Use Cardano's formula for one real root
                    u = (-q / 2 + math.sqrt(-disc / 108)) ** (1/3)
                    λ1 = u - p / (3 * u) - a / 3
                
                # Calculate complex conjugate pair
                # For cubic with one real root, the other two are: λ1/2 ± i·√(3)/2·(u + p/(3u))
                Re = -a / 3 - λ1 / 2
                Im = math.sqrt(3) / 2 * abs(u - p / (3 * u)) if p != 0 else 0
                λ2 = complex(Re, Im)
                λ3 = complex(Re, -Im)
                
                return [λ1, λ2, λ3]
        
        # For larger matrices, use a more sophisticated method
        raise MatPyNotImplementedError(
            "Eigenvalue calculation for matrices larger than 3x3",
            "Eigenvalue calculation for matrices larger than 3x3 requires "
            "numerical methods. Consider using NumPy: numpy.linalg.eig()"
        )
    
    @staticmethod
    def eigenvectors(A: Matrix) -> List[List[float]]:
        """
        Calculate eigenvectors of the matrix.
        
        Note: This is a simplified implementation for small matrices.
        For production use with larger matrices, consider using NumPy.
        
        Args:
            matrix: The matrix to calculate eigenvectors for
        
        Returns:
            List of eigenvectors corresponding to eigenvalues.
            Each eigenvector is normalized.
        
        Raises:
            ValueError: If matrix is not square
            NotImplementedError: For matrices larger than 3x3
        
        Example:
            >>> A = Matrix(2, 2, [[4, 2], [1, 3]])
            >>> eigenvecs = Matrix.eigenvectors(A)
            >>> # Returns list of 2D eigenvectors
        """
        require_square(A, "Eigenvectors")
        
        import math
        
        # Get eigenvalues first
        eigenvals = Matrix.eigenvalues(A)
        eigenvecs = []
        
        # For 1x1 matrix
        if A.rows == 1: return [[1.0]]
        
        # For 2x2 matrix
        if A.rows == 2:
            (a, b), (c, d) = A.data
            
            for λ in eigenvals:
                # Solve (A - λI)v = 0
                # [a-λ, b  ] [v1]   [0]
                # [c,   d-λ] [v2] = [0]
                
                if abs(b) > DEFAULT_TOLERANCE:
                    # Use first row: (a-λ)v1 + b·v2 = 0  =>  v2 = -(a-λ)/b · v1
                    v1 = 1.0
                    v2 = -(a - λ) / b
                elif abs(c) > 1e-10:
                    # Use second row: c·v1 + (d-λ)v2 = 0  =>  v1 = -(d-λ)/c · v2
                    v2 = 1.0
                    v1 = -(d - λ) / c
                else:
                    # Diagonal matrix - use standard basis
                    if abs(a - λ) < abs(d - λ):
                        v1, v2 = 1.0, 0.0
                    else:
                        v1, v2 = 0.0, 1.0
                
                # Normalize the eigenvector
                # Handle complex eigenvectors
                if isinstance(v1, complex) or isinstance(v2, complex):
                    # For complex vectors, use abs() which works on complex numbers
                    norm = abs(complex(v1)) ** 2 + abs(complex(v2)) ** 2
                    norm = math.sqrt(norm.real if isinstance(norm, complex) else norm)
                else:
                    norm = math.sqrt(v1 * v1 + v2 * v2)
                eigenvecs.append([v1 / norm, v2 / norm])
            
            return eigenvecs
        
        # For 3x3 matrix
        if A.rows == 3:
            for λ in eigenvals:
                # Solve (A - λI)v = 0 by finding null space
                # Construct (A - λI)
                A_minus_λ = [
                    [A.data[i][j] - (λ if i == j else 0)
                     for j in range(3)]
                    for i in range(3)
                ]
                
                # Find a non-trivial solution using cross product method
                # Take two rows and compute their cross product
                row1 = A_minus_λ[0]
                row2 = A_minus_λ[1]
                
                # Cross product gives a vector perpendicular to both rows
                v1 = row1[1] * row2[2] - row1[2] * row2[1]
                v2 = row1[2] * row2[0] - row1[0] * row2[2]
                v3 = row1[0] * row2[1] - row1[1] * row2[0]
                
                # If cross product is zero, try different rows
                # Handle complex eigenvectors
                norm_sq = abs(v1)**2 + abs(v2)**2 + abs(v3)**2
                norm = math.sqrt(norm_sq.real if isinstance(norm_sq, complex) else norm_sq)
                if norm < 1e-10:
                    row1 = A_minus_λ[0]
                    row2 = A_minus_λ[2]
                    v1 = row1[1] * row2[2] - row1[2] * row2[1]
                    v2 = row1[2] * row2[0] - row1[0] * row2[2]
                    v3 = row1[0] * row2[1] - row1[1] * row2[0]
                    norm_sq = abs(v1)**2 + abs(v2)**2 + abs(v3)**2
                    norm = math.sqrt(norm_sq.real if isinstance(norm_sq, complex) else norm_sq)
                
                if norm < 1e-10:
                    row1 = A_minus_λ[1]
                    row2 = A_minus_λ[2]
                    v1 = row1[1] * row2[2] - row1[2] * row2[1]
                    v2 = row1[2] * row2[0] - row1[0] * row2[2]
                    v3 = row1[0] * row2[1] - row1[1] * row2[0]
                    norm_sq = abs(v1)**2 + abs(v2)**2 + abs(v3)**2
                    norm = math.sqrt(norm_sq.real if isinstance(norm_sq, complex) else norm_sq)
                
                # Normalize
                if norm > 1e-10:
                    eigenvecs.append([v1 / norm, v2 / norm, v3 / norm])
                else:
                    # Fallback: use a standard basis vector
                    eigenvecs.append([1.0, 0.0, 0.0])
            
            return eigenvecs
        
        # For larger matrices
        raise MatPyNotImplementedError(
            "Eigenvector calculation for matrices larger than 3x3",
            "Eigenvector calculation for matrices larger than 3x3 requires "
            "numerical methods. Consider using NumPy: numpy.linalg.eig()"
        )