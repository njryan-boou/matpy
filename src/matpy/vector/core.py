from __future__ import annotations
import math

from ..error import IndexError as MatPyIndexError
from ..core import validate, utils


class Vector:
    """
    An n-dimensional vector class with support for common vector operations and Python dunder methods.
    
    Attributes:
        components (tuple): The components of the vector
    
    Note:
        For convenience, 2D and 3D vectors can access components via x, y, z properties.
    """
    
    # ==================== Initialization ====================
    
    def __init__(self, *args) -> None:
        """
        Initialize an n-dimensional vector.
        
        Args:
            *args: Variable number of components. If no arguments, creates a 3D zero vector.
        
        Examples:
            >>> Vector()           # 3D zero vector (0, 0, 0)
            >>> Vector(1, 2)       # 2D vector (1, 2)
            >>> Vector(1, 2, 3)    # 3D vector (1, 2, 3)
            >>> Vector(1, 2, 3, 4) # 4D vector (1, 2, 3, 4)
        """
        if len(args) == 0:
            # Default to 3D zero vector for backwards compatibility
            self.components = (0.0, 0.0, 0.0)
        else:
            self.components = tuple(float(x) for x in args)
    
    # Properties for backwards compatibility with 2D/3D vectors
    @property
    def x(self) -> float:
        """Get x-component (first component)."""
        return self.components[0] if len(self.components) > 0 else 0.0
    
    @property
    def y(self) -> float:
        """Get y-component (second component)."""
        return self.components[1] if len(self.components) > 1 else 0.0
    
    @property
    def z(self) -> float:
        """Get z-component (third component)."""
        return self.components[2] if len(self.components) > 2 else 0.0
    
    # ==================== String Representation ====================
    
    def __str__(self) -> str:
        """Return informal string representation: (c1, c2, ...)"""
        return f"({', '.join(str(c) for c in self.components)})"
    
    def __repr__(self) -> str:
        """Return official string representation: <c1, c2, ...>"""
        return f"<{', '.join(str(c) for c in self.components)}>"
    
    def __format__(self, format_spec: str) -> str:
        """Return custom formatted string representation."""
        return f"({', '.join(format(c, format_spec) for c in self.components)})"
    
    # ==================== Arithmetic Operators ====================
    
    def __add__(self, other: Vector) -> Vector:
        """Add two vectors: v1 + v2"""
        validate.validate_dimensions_match(
            len(self.components),
            len(other.components),
            "vector addition"
        )
        return Vector(*(a + b for a, b in zip(self.components, other.components)))
    
    def __radd__(self, other: Vector) -> Vector:
        """Reflected addition: other + v1"""
        return self.__add__(other)
    
    def __sub__(self, other: Vector) -> Vector:
        """Subtract two vectors: v1 - v2"""
        validate.validate_dimensions_match(
            len(self.components),
            len(other.components),
            "vector subtraction"
        )
        return Vector(*(a - b for a, b in zip(self.components, other.components)))
    
    def __mul__(self, scalar: float) -> Vector:
        """Multiply vector by scalar: v * scalar"""
        return Vector(*(c * scalar for c in self.components))
    
    def __rmul__(self, scalar: float) -> Vector:
        """Reflected multiplication: scalar * v"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> Vector:
        """Divide vector by scalar: v / scalar"""
        return Vector(*(c / scalar for c in self.components))
    
    # ==================== Unary Operators ====================
    
    def __neg__(self) -> Vector:
        """Negate vector: -v"""
        return Vector(*(-c for c in self.components))
    
    def __pos__(self) -> Vector:
        """Unary plus: +v"""
        return Vector(*self.components)
    
    def __abs__(self) -> float:
        """Return magnitude of vector: abs(v)"""
        return self.magnitude()
    
    # ==================== Type Conversion ====================
    
    def __bool__(self) -> bool:
        """Return True if vector is non-zero."""
        return any(c != 0 for c in self.components)
    
    def __round__(self, n: int = 0) -> Vector:
        """Round vector components to n digits."""
        return Vector(*(round(c, n) for c in self.components))
    
    # ==================== Container/Sequence Methods ====================
    
    def __len__(self) -> int:
        """Return the number of components."""
        return len(self.components)
    
    def __getitem__(self, index: int) -> float:
        """Get component by index: v[0], v[1], v[2], ..."""
        validate.validate_index(index, len(self.components), "vector index")
        return self.components[index]
    
    def __iter__(self):
        """Iterate over vector components."""
        return iter(self.components)
    
    def __contains__(self, value: float) -> bool:
        """Check if value exists in any component: value in v"""
        return value in self.components
    
    # ==================== Copying ====================
    
    def __copy__(self) -> Vector:
        """Create a shallow copy of the vector."""
        return Vector(*self.components)
    
    # ==================== Vector Operations ====================
    
    def magnitude(self) -> float:
        """Calculate and return the magnitude (length) of the vector."""
        return math.sqrt(sum(c ** 2 for c in self.components))
    
    def dot(self, other: Vector) -> float:
        """
        Calculate dot product with another vector.
        
        Raises:
            DimensionError: If vectors have different dimensions
        """
        validate.validate_dimensions_match(
            len(self.components),
            len(other.components),
            "dot product"
        )
        return sum(a * b for a, b in zip(self.components, other.components))
    
    def cross(self, other: Vector) -> Vector:
        """
        Calculate cross product with another vector.
        
        Note:
            Cross product is only defined for 3D vectors.
        
        Raises:
            DimensionError: If either vector is not 3D
        """
        validate.validate_vector_dimension(len(self.components), 3, "cross product (first vector)")
        validate.validate_vector_dimension(len(other.components), 3, "cross product (second vector)")
        return Vector(
            self.components[1] * other.components[2] - self.components[2] * other.components[1],
            self.components[2] * other.components[0] - self.components[0] * other.components[2],
            self.components[0] * other.components[1] - self.components[1] * other.components[0]
        )
    
    def normalize(self) -> Vector:
        """Return a normalized (unit) vector in the same direction."""
        mag = self.magnitude()
        if validate.approx_zero(mag):
            return Vector(*self.components)
        return self / mag