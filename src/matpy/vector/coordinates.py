"""
Coordinate system conversion module for vectors.

This module provides classes for converting between different coordinate systems:
- 2D: Cartesian, Polar, Complex
- 3D: Cartesian, Spherical, Cylindrical
"""

from __future__ import annotations
import math
import cmath
from typing import Tuple, Union

from .core import Vector
from ..error import DimensionError, ValidationError
from ..core import validate


class VectorCoordinates:
    """
    A class for converting vectors between different coordinate systems.
    
    Supports:
    - 2D: Cartesian (x, y), Polar (r, θ), Complex (a + bi)
    - 3D: Cartesian (x, y, z), Spherical (r, θ, φ), Cylindrical (ρ, φ, z)
    """
    
    def __init__(self, vector: Vector):
        """
        Initialize coordinate converter with a vector.
        
        Args:
            vector: A 2D or 3D vector
        
        Raises:
            DimensionError: If vector is not 2D or 3D
        """
        if len(vector.components) not in (2, 3):
            validate.validate_vector_dimension(
                len(vector.components),
                (2, 3),
                "coordinate conversion"
            )
        self.vector = vector
        self.dimension = len(vector.components)
    
    # ==================== 2D Coordinate Systems ====================
    
    def to_polar(self) -> Tuple[float, float]:
        """
        Convert 2D Cartesian to Polar coordinates.
        
        Returns:
            Tuple (r, theta) where:
                r: radius (distance from origin)
                theta: angle in radians from positive x-axis
        
        Raises:
            DimensionError: If vector is not 2D
        
        Example:
            >>> v = Vector(1, 1)
            >>> coords = VectorCoordinates(v)
            >>> r, theta = coords.to_polar()
            >>> # r ≈ 1.414, theta ≈ 0.785 (π/4)
        """
        validate.validate_vector_dimension(self.dimension, 2, "polar conversion")
        
        x, y = self.vector.x, self.vector.y
        r = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x)
        
        return (r, theta)
    
    @staticmethod
    def from_polar(r: float, theta: float) -> Vector:
        """
        Create a 2D vector from Polar coordinates.
        
        Args:
            r: Radius (distance from origin)
            theta: Angle in radians from positive x-axis
        
        Returns:
            2D Vector in Cartesian coordinates
        
        Example:
            >>> v = VectorCoordinates.from_polar(1.414, math.pi/4)
            >>> # v ≈ Vector(1, 1)
        """
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return Vector(x, y)
    
    def to_complex(self) -> complex:
        """
        Convert 2D Cartesian to Complex number.
        
        Returns:
            Complex number (x + yi)
        
        Raises:
            DimensionError: If vector is not 2D
        
        Example:
            >>> v = Vector(3, 4)
            >>> coords = VectorCoordinates(v)
            >>> c = coords.to_complex()
            >>> # c = 3 + 4j
        """
        if self.dimension != 2:
            raise DimensionError(
                self.dimension,
                2,
                "complex conversion",
                "Complex coordinates require 2D vector"
            )
        
        return complex(self.vector.x, self.vector.y)
    
    @staticmethod
    def from_complex(c: complex) -> Vector:
        """
        Create a 2D vector from Complex number.
        
        Args:
            c: Complex number (a + bi)
        
        Returns:
            2D Vector with x=a, y=b
        
        Example:
            >>> v = VectorCoordinates.from_complex(3 + 4j)
            >>> # v = Vector(3, 4)
        """
        return Vector(c.real, c.imag)
    
    def to_polar_complex(self) -> Tuple[float, float]:
        """
        Convert 2D vector to polar form using complex representation.
        
        Returns:
            Tuple (magnitude, phase) where:
                magnitude: |z| (absolute value)
                phase: arg(z) (angle in radians)
        
        Raises:
            DimensionError: If vector is not 2D
        """
        if self.dimension != 2:
            raise DimensionError(
                self.dimension,
                2,
                "polar complex conversion",
                "Polar complex requires 2D vector"
            )
        
        c = self.to_complex()
        return (abs(c), cmath.phase(c))
    
    # ==================== 3D Coordinate Systems ====================
    
    def to_spherical(self) -> Tuple[float, float, float]:
        """
        Convert 3D Cartesian to Spherical coordinates.
        
        Returns:
            Tuple (r, theta, phi) where:
                r: radial distance from origin
                theta: azimuthal angle in radians (angle in xy-plane from +x axis)
                phi: polar angle in radians (angle from +z axis)
        
        Raises:
            DimensionError: If vector is not 3D
        
        Example:
            >>> v = Vector(1, 1, 1)
            >>> coords = VectorCoordinates(v)
            >>> r, theta, phi = coords.to_spherical()
        
        Note:
            Follows physics convention (ISO 31-11):
            - r ≥ 0
            - θ ∈ [0, 2π)
            - φ ∈ [0, π]
        """
        if self.dimension != 3:
            raise DimensionError(
                self.dimension,
                3,
                "spherical conversion",
                "Spherical coordinates require 3D vector"
            )
        
        x, y, z = self.vector.x, self.vector.y, self.vector.z
        
        r = math.sqrt(x**2 + y**2 + z**2)
        
        if r == 0:
            return (0.0, 0.0, 0.0)
        
        theta = math.atan2(y, x)  # Azimuthal angle
        phi = math.acos(z / r)     # Polar angle from +z axis
        
        return (r, theta, phi)
    
    @staticmethod
    def from_spherical(r: float, theta: float, phi: float) -> Vector:
        """
        Create a 3D vector from Spherical coordinates.
        
        Args:
            r: Radial distance from origin
            theta: Azimuthal angle in radians (angle in xy-plane from +x axis)
            phi: Polar angle in radians (angle from +z axis)
        
        Returns:
            3D Vector in Cartesian coordinates
        
        Example:
            >>> v = VectorCoordinates.from_spherical(1, 0, math.pi/2)
            >>> # v ≈ Vector(1, 0, 0)
        
        Note:
            Uses physics convention (ISO 31-11):
            - r ≥ 0
            - θ ∈ [0, 2π)
            - φ ∈ [0, π]
        """
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        
        return Vector(x, y, z)
    
    def to_cylindrical(self) -> Tuple[float, float, float]:
        """
        Convert 3D Cartesian to Cylindrical coordinates.
        
        Returns:
            Tuple (rho, phi, z) where:
                rho: radial distance from z-axis
                phi: azimuthal angle in radians (angle in xy-plane from +x axis)
                z: height (same as Cartesian z)
        
        Raises:
            DimensionError: If vector is not 3D
        
        Example:
            >>> v = Vector(1, 1, 2)
            >>> coords = VectorCoordinates(v)
            >>> rho, phi, z = coords.to_cylindrical()
        
        Note:
            - ρ ≥ 0
            - φ ∈ [0, 2π)
            - z ∈ ℝ
        """
        if self.dimension != 3:
            raise DimensionError(
                self.dimension,
                3,
                "cylindrical conversion",
                "Cylindrical coordinates require 3D vector"
            )
        
        x, y, z = self.vector.x, self.vector.y, self.vector.z
        
        rho = math.sqrt(x**2 + y**2)
        phi = math.atan2(y, x)
        
        return (rho, phi, z)
    
    @staticmethod
    def from_cylindrical(rho: float, phi: float, z: float) -> Vector:
        """
        Create a 3D vector from Cylindrical coordinates.
        
        Args:
            rho: Radial distance from z-axis
            phi: Azimuthal angle in radians (angle in xy-plane from +x axis)
            z: Height
        
        Returns:
            3D Vector in Cartesian coordinates
        
        Example:
            >>> v = VectorCoordinates.from_cylindrical(1, math.pi/4, 2)
            >>> # v ≈ Vector(0.707, 0.707, 2)
        """
        x = rho * math.cos(phi)
        y = rho * math.sin(phi)
        
        return Vector(x, y, z)
    
    # ==================== Conversion Helpers ====================
    
    def convert(self, to_system: str) -> Union[Vector, Tuple, complex]:
        """
        Convert vector to specified coordinate system.
        
        Args:
            to_system: Target system - 'polar', 'complex', 'spherical', 'cylindrical'
        
        Returns:
            Converted coordinates in the specified system
        
        Raises:
            ValidationError: If system is invalid
            DimensionError: If conversion not possible for vector dimension
        
        Example:
            >>> v = Vector(1, 1)
            >>> coords = VectorCoordinates(v)
            >>> polar = coords.convert('polar')
            >>> complex_num = coords.convert('complex')
        """
        system = to_system.lower()
        
        if system == 'polar':
            return self.to_polar()
        elif system == 'complex':
            return self.to_complex()
        elif system == 'spherical':
            return self.to_spherical()
        elif system == 'cylindrical':
            return self.to_cylindrical()
        else:
            raise ValidationError(
                f"Invalid coordinate system: {to_system}. "
                f"Valid options: 'polar', 'complex', 'spherical', 'cylindrical'"
            )
    
    @staticmethod
    def convert_from(system: str, *args) -> Vector:
        """
        Create vector from specified coordinate system.
        
        Args:
            system: Source system - 'polar', 'complex', 'spherical', 'cylindrical'
            *args: Coordinates in the specified system
        
        Returns:
            Vector in Cartesian coordinates
        
        Raises:
            ValidationError: If system is invalid
        
        Example:
            >>> v = VectorCoordinates.convert_from('polar', 1.414, math.pi/4)
            >>> v2 = VectorCoordinates.convert_from('complex', 3+4j)
            >>> v3 = VectorCoordinates.convert_from('spherical', 1, 0, math.pi/2)
        """
        system = system.lower()
        
        if system == 'polar':
            if len(args) != 2:
                raise ValidationError("Polar conversion requires 2 arguments: (r, theta)")
            return VectorCoordinates.from_polar(*args)
        
        elif system == 'complex':
            if len(args) != 1:
                raise ValidationError("Complex conversion requires 1 argument: (complex_number)")
            return VectorCoordinates.from_complex(args[0])
        
        elif system == 'spherical':
            if len(args) != 3:
                raise ValidationError("Spherical conversion requires 3 arguments: (r, theta, phi)")
            return VectorCoordinates.from_spherical(*args)
        
        elif system == 'cylindrical':
            if len(args) != 3:
                raise ValidationError("Cylindrical conversion requires 3 arguments: (rho, phi, z)")
            return VectorCoordinates.from_cylindrical(*args)
        
        else:
            raise ValidationError(
                f"Invalid coordinate system: {system}. "
                f"Valid options: 'polar', 'complex', 'spherical', 'cylindrical'"
            )
    
    def __str__(self) -> str:
        """String representation showing all available conversions."""
        if self.dimension == 2:
            r, theta = self.to_polar()
            c = self.to_complex()
            return (
                f"2D Vector Coordinates:\n"
                f"  Cartesian: ({self.vector.x}, {self.vector.y})\n"
                f"  Polar: (r={r:.4f}, θ={theta:.4f} rad)\n"
                f"  Complex: {c}"
            )
        else:  # 3D
            r, theta, phi = self.to_spherical()
            rho, phi_cyl, z = self.to_cylindrical()
            return (
                f"3D Vector Coordinates:\n"
                f"  Cartesian: ({self.vector.x}, {self.vector.y}, {self.vector.z})\n"
                f"  Spherical: (r={r:.4f}, θ={theta:.4f}, φ={phi:.4f})\n"
                f"  Cylindrical: (ρ={rho:.4f}, φ={phi_cyl:.4f}, z={z:.4f})"
            )
    
    def __repr__(self) -> str:
        """Official string representation."""
        return f"VectorCoordinates({self.dimension}D: {self.vector})"


# ==================== Convenience Functions ====================

def to_polar(vector: Vector) -> Tuple[float, float]:
    """Convert 2D vector to polar coordinates (r, θ)."""
    return VectorCoordinates(vector).to_polar()


def from_polar(r: float, theta: float) -> Vector:
    """Create 2D vector from polar coordinates."""
    return VectorCoordinates.from_polar(r, theta)


def to_complex(vector: Vector) -> complex:
    """Convert 2D vector to complex number."""
    return VectorCoordinates(vector).to_complex()


def from_complex(c: complex) -> Vector:
    """Create 2D vector from complex number."""
    return VectorCoordinates.from_complex(c)


def to_spherical(vector: Vector) -> Tuple[float, float, float]:
    """Convert 3D vector to spherical coordinates (r, θ, φ)."""
    return VectorCoordinates(vector).to_spherical()


def from_spherical(r: float, theta: float, phi: float) -> Vector:
    """Create 3D vector from spherical coordinates."""
    return VectorCoordinates.from_spherical(r, theta, phi)


def to_cylindrical(vector: Vector) -> Tuple[float, float, float]:
    """Convert 3D vector to cylindrical coordinates (ρ, φ, z)."""
    return VectorCoordinates(vector).to_cylindrical()


def from_cylindrical(rho: float, phi: float, z: float) -> Vector:
    """Create 3D vector from cylindrical coordinates."""
    return VectorCoordinates.from_cylindrical(rho, phi, z)
