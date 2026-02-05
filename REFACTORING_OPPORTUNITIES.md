"""
REFACTORING OPPORTUNITIES - Using validate.py and utils.py

This document identifies places in the codebase where we can replace
existing code with functions from validate.py and utils.py.
"""

# ==================== MATRIX CORE.PY ====================

"""
CURRENT CODE (matrix/core.py line 36-56):
    def __init__(self, rows: int, cols: int, data: List[List[float]] = None):
        self.rows = rows
        self.cols = cols
        
        if data is None:
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]
        else:
            if len(data) != rows or any(len(row) != cols for row in data):
                raise ValidationError("Data dimensions do not match specified rows and columns.")
            self.data = data

REFACTORED WITH validate.py:
    def __init__(self, rows: int, cols: int, data: List[List[float]] = None):
        from ..core.validate import validate_matrix_dimensions, validate_data_shape
        
        validate_matrix_dimensions(rows, cols)
        self.rows = rows
        self.cols = cols
        
        if data is None:
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]
        else:
            validate_data_shape(data, rows, cols, "data")
            self.data = data
"""

"""
CURRENT CODE (matrix/core.py line 78-88):
    def __add__(self, other: Matrix) -> Matrix:
        if self.rows != other.rows or self.cols != other.cols:
            raise ShapeError(
                (self.rows, self.cols),
                (other.rows, other.cols),
                "addition"
            )

REFACTORED WITH validate.py:
    def __add__(self, other: Matrix) -> Matrix:
        from ..core.validate import validate_same_shape
        
        validate_same_shape(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "addition"
        )
"""

"""
CURRENT CODE (matrix/core.py - determinant, inverse, etc.):
    if not self.is_square():
        raise NotSquareError((self.rows, self.cols), "determinant")

REFACTORED WITH validate.py:
    from ..core.validate import validate_square_matrix
    
    validate_square_matrix(self.rows, self.cols, "determinant")
"""

"""
CURRENT CODE (matrix/core.py line 371-372):
    if len(value) != self.cols:
        raise ValidationError(f"Row must have {self.cols} elements.")

REFACTORED WITH validate.py:
    from ..core.validate import validate_dimensions_match
    
    validate_dimensions_match(len(value), self.cols, "row assignment")
"""

# ==================== VECTOR CORE.PY ====================

"""
CURRENT CODE (vector/core.py line 74-80):
    def __add__(self, other: Vector) -> Vector:
        from ..error import DimensionError
        if len(self.components) != len(other.components):
            raise DimensionError(
                len(self.components),
                len(other.components),
                "vector addition"
            )

REFACTORED WITH validate.py:
    def __add__(self, other: Vector) -> Vector:
        from ..core.validate import validate_dimensions_match
        
        validate_dimensions_match(
            len(self.components),
            len(other.components),
            "vector addition"
        )
"""

"""
CURRENT CODE (vector/core.py - dot product):
    def dot(self, other: Vector) -> float:
        from ..error import DimensionError
        if len(self.components) != len(other.components):
            raise DimensionError(...)

REFACTORED WITH validate.py:
    def dot(self, other: Vector) -> float:
        from ..core.validate import validate_dimensions_match
        
        validate_dimensions_match(
            len(self.components),
            len(other.components),
            "dot product"
        )
"""

"""
CURRENT CODE (vector/core.py - cross product):
    def cross(self, other: Vector) -> Vector:
        from ..error import DimensionError
        if len(self.components) != 3 or len(other.components) != 3:
            raise DimensionError(...)

REFACTORED WITH validate.py:
    def cross(self, other: Vector) -> Vector:
        from ..core.validate import validate_vector_dimension
        
        validate_vector_dimension(len(self.components), 3, "cross product")
        validate_vector_dimension(len(other.components), 3, "cross product")
"""

# ==================== VECTOR COORDINATES.PY ====================

"""
CURRENT CODE (vector/coordinates.py line 26-34):
    def __init__(self, vector: Vector):
        if len(vector.components) not in (2, 3):
            raise DimensionError(
                len(vector.components),
                "2 or 3",
                "coordinate conversion",
                "Coordinate conversion only supports 2D and 3D vectors"
            )

REFACTORED WITH validate.py:
    def __init__(self, vector: Vector):
        from ..core.validate import validate_vector_dimension
        
        validate_vector_dimension(
            len(vector.components),
            (2, 3),
            "coordinate conversion"
        )
"""

# ==================== MATRIX OPS.PY ====================

"""
CURRENT CODE (matrix/ops.py - from_rows):
    def from_rows(rows: List[List[float]]) -> Matrix:
        if not rows:
            raise ValueError("Cannot create matrix from empty list")
        
        num_rows = len(rows)
        num_cols = len(rows[0])
        
        if any(len(row) != num_cols for row in rows):
            raise ValueError("All rows must have the same length")

REFACTORED WITH validate.py:
    def from_rows(rows: List[List[float]]) -> Matrix:
        from ..core.validate import validate_rectangular_data
        
        num_rows, num_cols = validate_rectangular_data(rows, "rows")
        return Matrix(num_rows, num_cols, rows)
"""

"""
CURRENT CODE (matrix/ops.py - hadamard_product):
    if m1.rows != m2.rows or m1.cols != m2.cols:
        raise ShapeError(...)

REFACTORED WITH validate.py:
    from ..core.validate import validate_same_shape
    
    validate_same_shape(
        (m1.rows, m1.cols),
        (m2.rows, m2.cols),
        "Hadamard product"
    )
"""

# ==================== MATRIX SOLVE.PY ====================

"""
CURRENT CODE (matrix/solve.py - solve_linear_system):
    if not A.is_square():
        from ..error import NotSquareError
        raise NotSquareError((A.rows, A.cols), "solve linear system")
    
    if len(b) != A.rows:
        raise ShapeError(...)

REFACTORED WITH validate.py:
    from ..core.validate import validate_square_matrix, validate_dimensions_match
    
    validate_square_matrix(A.rows, A.cols, "solve linear system")
    validate_dimensions_match(len(b), A.rows, "system solve (b vector)")
"""

"""
CURRENT CODE (matrix/solve.py - various places checking det == 0):
    if abs(det_A) < 1e-10:
        raise SingularMatrixError("Cannot solve system (determinant is zero)")

REFACTORED WITH utils.py:
    from ..core.utils import approx_zero
    
    if approx_zero(det_A):
        raise SingularMatrixError("Cannot solve system (determinant is zero)")
"""

# ==================== VECTOR OPS.PY ====================

"""
CURRENT CODE (vector/ops.py - angle_between):
    mag_v1 = abs(v1)
    mag_v2 = abs(v2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0

REFACTORED WITH utils.py:
    from ..core.utils import approx_zero
    
    mag_v1 = abs(v1)
    mag_v2 = abs(v2)
    
    if approx_zero(mag_v1) or approx_zero(mag_v2):
        return 0.0
"""

"""
CURRENT CODE (vector/ops.py - projection):
    mag_v2 = abs(v2)
    if mag_v2 == 0:
        return Vector(*([0] * len(v1.components)))

REFACTORED WITH utils.py and validate.py:
    from ..core.utils import approx_zero
    from ..core.validate import validate_dimensions_match
    
    validate_dimensions_match(
        len(v1.components),
        len(v2.components),
        "projection"
    )
    
    mag_v2 = abs(v2)
    if approx_zero(mag_v2):
        return Vector(*([0] * len(v1.components)))
"""

"""
CURRENT CODE (vector/ops.py - is_parallel):
    mag1 = abs(v1)
    mag2 = abs(v2)
    
    if mag1 < tolerance or mag2 < tolerance:
        return True

REFACTORED WITH utils.py:
    from ..core.utils import approx_zero
    
    mag1 = abs(v1)
    mag2 = abs(v2)
    
    if approx_zero(mag1, tolerance) or approx_zero(mag2, tolerance):
        return True
"""

# ==================== STRING FORMATTING ====================

"""
CURRENT CODE (matrix/core.py - __str__):
    def __str__(self) -> str:
        return '\n'.join(['\t'.join(map(str, row)) for row in self.data])

COULD USE utils.py:
    def __str__(self) -> str:
        from ..core.utils import format_matrix_string
        return format_matrix_string(self.data)
"""

"""
CURRENT CODE (vector/core.py - __str__):
    def __str__(self) -> str:
        return f"({', '.join(str(c) for c in self.components)})"

COULD USE utils.py:
    def __str__(self) -> str:
        from ..core.utils import format_vector_string
        return format_vector_string(self.components)
"""

# ==================== MATRIX OPERATIONS ====================

"""
CURRENT CODE (matrix/core.py - transpose):
    result_data = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]

COULD USE utils.py:
    from ..core.utils import transpose_list
    result_data = transpose_list(self.data)
"""

"""
CURRENT CODE (matrix/core.py - __copy__):
    data_copy = [row[:] for row in self.data]

COULD USE utils.py:
    from ..core.utils import deep_copy_2d
    data_copy = deep_copy_2d(self.data)
"""

# ==================== COMPARISON OPERATIONS ====================

"""
CURRENT CODE (matrix/core.py - __eq__):
    for i in range(self.rows):
        for j in range(self.cols):
            if abs(self.data[i][j] - other.data[i][j]) > 1e-10:
                return False

COULD USE utils.py:
    from ..core.utils import is_close
    
    for i in range(self.rows):
        for j in range(self.cols):
            if not is_close(self.data[i][j], other.data[i][j]):
                return False
"""

"""
CURRENT CODE (matrix/core.py - is_symmetric):
    if abs(self.data[i][j] - self.data[j][i]) > 1e-10:
        return False

COULD USE utils.py:
    from ..core.utils import is_close
    
    if not is_close(self.data[i][j], self.data[j][i]):
        return False
"""

# ==================== INDEX VALIDATION ====================

"""
CURRENT CODE (matrix/core.py - __getitem__):
    row, col = key
    if not (0 <= row < self.rows) or not (0 <= col < self.cols):
        raise MatPyIndexError(...)

COULD USE validate.py:
    from ..core.validate import validate_matrix_index
    
    row, col = key
    validate_matrix_index(row, col, self.rows, self.cols)
"""

"""
CURRENT CODE (vector/core.py - __getitem__):
    try:
        return self.components[index]
    except IndexError:
        raise MatPyIndexError(...)

COULD USE validate.py:
    from ..core.validate import validate_index
    
    validate_index(index, len(self.components), "index")
    return self.components[index]
"""

# ==================== SUMMARY ====================

"""
KEY OPPORTUNITIES:

1. VALIDATION REPLACEMENTS:
   - Replace dimension checking with validate_dimensions_match()
   - Replace square matrix checks with validate_square_matrix()
   - Replace shape matching with validate_same_shape()
   - Replace rectangular data checks with validate_rectangular_data()
   - Replace index bounds checks with validate_index() and validate_matrix_index()

2. UTILITY REPLACEMENTS:
   - Replace zero checks (== 0) with approx_zero()
   - Replace equality checks (abs(a-b) < tol) with is_close()
   - Replace list transposing with transpose_list()
   - Replace 2D list copying with deep_copy_2d()
   - Replace string formatting with format_matrix_string() and format_vector_string()

3. BENEFITS:
   - More consistent error messages
   - Better tolerance handling (no hard-coded 1e-10)
   - Cleaner, more readable code
   - Centralized validation logic
   - Easier to maintain and test

4. FILES TO UPDATE:
   - src/matpy/matrix/core.py (highest priority)
   - src/matpy/vector/core.py (highest priority)
   - src/matpy/matrix/ops.py
   - src/matpy/matrix/solve.py
   - src/matpy/vector/ops.py
   - src/matpy/vector/coordinates.py
"""
