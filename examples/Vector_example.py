"""
Vector Examples - Showcasing core.py and ops.py functionality
"""
import sys
import math
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matpy.vector.core import Vector
from matpy.vector import ops

print("=" * 60)
print("CORE.PY - Vector Class with Dunder Methods")
print("=" * 60)

# Creation and Basic String Representation
print("\n1. Creating Vectors (__init__, __str__, __repr__)")
v1 = Vector(3, 4, 5)
v2 = Vector(1, 2, 3)
print(f"v1 = {v1}")  # Uses __str__
print(f"v1 repr: {repr(v1)}")  # Uses __repr__

# Arithmetic Operations
print("\n2. Arithmetic Operations (__add__, __sub__, __mul__, __truediv__, __neg__)")
print(f"v1 + v2 = {v1 + v2}")  # __add__
print(f"v1 - v2 = {v1 - v2}")  # __sub__
print(f"v1 * 2 = {v1 * 2}")    # __mul__
print(f"2 * v1 = {2 * v1}")    # __rmul__
print(f"v1 / 2 = {v1 / 2}")    # __truediv__
print(f"-v1 = {-v1}")          # __neg__

# Vector Operations
print("\n3. Vector-Specific Operations (dot, cross, normalize)")
print(f"v1.dot(v2) = {v1.dot(v2)}")
print(f"v1.cross(v2) = {v1.cross(v2)}")

# Indexing and Iteration
print("\n4. Indexing and Iteration (__getitem__, __iter__, __len__, __contains__)")
print(f"v1[0] = {v1[0]}, v1[1] = {v1[1]}, v1[2] = {v1[2]}")  # __getitem__
print(f"len(v1) = {len(v1)}")  # __len__
print(f"3 in v1: {3 in v1}")   # __contains__
print(f"Iterating: ", end="")
for component in v1:  # __iter__
    print(f"{component} ", end="")
print()

# Unary Operations
print("\n5. Unary Operations (__abs__, __round__)")
v3 = Vector(3.456, 4.789, 5.123)
print(f"v3 = {v3}")
print(f"round(v3, 1) = {round(v3, 1)}")  # __round__

# Formatting
print("\n6. Custom Formatting (__format__)")
print(f"v3 formatted to 2 decimals: {v3:.2f}")  # __format__

# Boolean Conversion
print("\n7. Boolean Conversion (__bool__)")
v_zero = Vector(0, 0, 0)
v_nonzero = Vector(1, 0, 0)
print(f"bool(Vector(0,0,0)) = {bool(v_zero)}")      # __bool__
print(f"bool(Vector(1,0,0)) = {bool(v_nonzero)}")   # __bool__

# Copying
print("\n8. Copying (__copy__)")
import copy
v_original = Vector(10, 20, 30)
v_copied = copy.copy(v_original)  # __copy__
print(f"Original: {v_original}, Copy: {v_copied}")
print(f"Same object? {v_original is v_copied}")

print("\n" + "=" * 60)
print("OPS.PY - Vector Operations Functions")
print("=" * 60)

# Magnitude
print("\n1. Magnitude")
v4 = Vector(3, 4, 0)
print(f"v4 = {v4}")
print(f"ops.magnitude(v4) = {ops.magnitude(v4)}")

# Normalization
print("\n2. Normalize")
v5 = Vector(5, 0, 0)
print(f"v5 = {v5}")
normalized = ops.normalize(v5)
print(f"ops.normalize(v5) = {normalized}")
print(f"magnitude after normalize = {ops.magnitude(normalized)}")

# Dot Product
print("\n3. Dot Product")
v6 = Vector(1, 0, 0)
v7 = Vector(0, 1, 0)
print(f"v6 = {v6}, v7 = {v7}")
print(f"ops.dot(v6, v7) = {ops.dot(v6, v7)}")

# Cross Product
print("\n4. Cross Product")
print(f"ops.cross(v6, v7) = {ops.cross(v6, v7)}")

# Angle Between
print("\n5. Angle Between Vectors")
v8 = Vector(1, 0, 0)
v9 = Vector(1, 1, 0)
angle_rad = ops.angle_between(v8, v9)
angle_deg = math.degrees(angle_rad)
print(f"v8 = {v8}, v9 = {v9}")
print(f"ops.angle_between(v8, v9) = {angle_rad:.4f} rad ({angle_deg:.2f}°)")

# Projection
print("\n6. Vector Projection")
v10 = Vector(3, 4, 0)
v11 = Vector(1, 0, 0)
proj = ops.projection(v10, v11)
print(f"v10 = {v10}, v11 = {v11}")
print(f"ops.projection(v10 onto v11) = {proj}")

# Rejection
print("\n7. Vector Rejection")
rej = ops.rejection(v10, v11)
print(f"ops.rejection(v10, v11) = {rej}")
print(f"Verify: projection + rejection = {Vector(proj.x + rej.x, proj.y + rej.y, proj.z + rej.z)}")

# Reflection
print("\n8. Vector Reflection")
v12 = Vector(1, -1, 0)
normal = Vector(0, 1, 0)
reflected = ops.reflect(v12, normal)
print(f"v12 = {v12}, normal = {normal}")
print(f"ops.reflect(v12, normal) = {reflected}")

# Linear Interpolation
print("\n9. Linear Interpolation (lerp)")
v13 = Vector(0, 0, 0)
v14 = Vector(10, 10, 10)
print(f"v13 = {v13}, v14 = {v14}")
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    lerp_result = ops.lerp(v13, v14, t)
    print(f"  t={t}: ops.lerp(v13, v14, {t}) = {lerp_result}")

print("\n" + "=" * 60)
print("COMBINED EXAMPLES - Real-World Applications")
print("=" * 60)

# Physics: Force vectors
print("\n1. Physics - Force Combination")
force1 = Vector(10, 0, 0)  # 10N to the right
force2 = Vector(0, 10, 0)  # 10N upward
total_force = force1 + force2
print(f"Force 1: {force1} N")
print(f"Force 2: {force2} N")
print(f"Total Force: {total_force} N")
print(f"Magnitude: {ops.magnitude(total_force):.2f} N")

# Graphics: Surface normal
print("\n2. Graphics - Surface Normal Calculation")
edge1 = Vector(1, 0, 0)
edge2 = Vector(0, 1, 0)
surface_normal = edge1.cross(edge2)
print(f"Edge 1: {edge1}")
print(f"Edge 2: {edge2}")
print(f"Surface Normal: {surface_normal}")

# Game Development: Moving object
print("\n3. Game Development - Smooth Movement")
start_pos = Vector(0, 0, 0)
end_pos = Vector(100, 50, 25)
print(f"Start: {start_pos}, End: {end_pos}")
print("Animation frames:")
for frame in range(0, 11, 2):
    t = frame / 10
    current_pos = ops.lerp(start_pos, end_pos, t)
    print(f"  Frame {frame}: {current_pos:.1f}")

print("\n" + "=" * 60)
