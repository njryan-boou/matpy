"""
Utility functions for matpy library.

This module provides general-purpose utility functions used throughout
the library for common operations, formatting, conversions, and helpers.
"""

from __future__ import annotations
from typing import List, Tuple, Any, Callable, Union
import math


# ==================== Math Utilities ====================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Value clamped to [min_val, max_val]
    
    Example:
        >>> clamp(5, 0, 10)
        5
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(15, 0, 10)
        10
    """
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value (t=0)
        b: End value (t=1)
        t: Interpolation parameter
    
    Returns:
        Interpolated value
    
    Example:
        >>> lerp(0, 10, 0.5)
        5.0
        >>> lerp(0, 10, 0.25)
        2.5
    """
    return a * (1 - t) + b * t


def sign(x: float) -> int:
    """
    Return the sign of a number.
    
    Args:
        x: Number to get sign of
    
    Returns:
        -1 if x < 0, 0 if x == 0, 1 if x > 0
    
    Example:
        >>> sign(5)
        1
        >>> sign(-3)
        -1
        >>> sign(0)
        0
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def degrees_to_radians(degrees: float) -> float:
    """
    Convert degrees to radians.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        Angle in radians
    
    Example:
        >>> degrees_to_radians(180)
        3.141592653589793
        >>> degrees_to_radians(90)
        1.5707963267948966
    """
    return degrees * math.pi / 180


def radians_to_degrees(radians: float) -> float:
    """
    Convert radians to degrees.
    
    Args:
        radians: Angle in radians
    
    Returns:
        Angle in degrees
    
    Example:
        >>> radians_to_degrees(math.pi)
        180.0
        >>> radians_to_degrees(math.pi / 2)
        90.0
    """
    return radians * 180 / math.pi


def factorial(n: int) -> int:
    """
    Calculate factorial of n.
    
    Args:
        n: Non-negative integer
    
    Returns:
        n! = n * (n-1) * ... * 2 * 1
    
    Example:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# ==================== List/Array Utilities ====================

def flatten(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a 2D list into a 1D list.
    
    Args:
        nested_list: 2D list to flatten
    
    Returns:
        Flattened 1D list
    
    Example:
        >>> flatten([[1, 2], [3, 4], [5, 6]])
        [1, 2, 3, 4, 5, 6]
    """
    return [item for row in nested_list for item in row]


def unflatten(flat_list: List[Any], cols: int) -> List[List[Any]]:
    """
    Convert a flat list into a 2D list with specified column count.
    
    Args:
        flat_list: 1D list to unflatten
        cols: Number of columns per row
    
    Returns:
        2D list
    
    Example:
        >>> unflatten([1, 2, 3, 4, 5, 6], 2)
        [[1, 2], [3, 4], [5, 6]]
    """
    return [flat_list[i:i+cols] for i in range(0, len(flat_list), cols)]


def transpose_list(data: List[List[Any]]) -> List[List[Any]]:
    """
    Transpose a 2D list (swap rows and columns).
    
    Args:
        data: 2D list to transpose
    
    Returns:
        Transposed 2D list
    
    Example:
        >>> transpose_list([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
    """
    if not data:
        return []
    return [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]


def deep_copy_2d(data: List[List[Any]]) -> List[List[Any]]:
    """
    Create a deep copy of a 2D list.
    
    Args:
        data: 2D list to copy
    
    Returns:
        Deep copy of the list
    
    Example:
        >>> original = [[1, 2], [3, 4]]
        >>> copy = deep_copy_2d(original)
        >>> copy[0][0] = 99
        >>> original[0][0]  # Still 1
        1
    """
    return [row[:] for row in data]


def all_same(items: List[Any]) -> bool:
    """
    Check if all items in a list are the same.
    
    Args:
        items: List to check
    
    Returns:
        True if all items are equal, False otherwise
    
    Example:
        >>> all_same([1, 1, 1, 1])
        True
        >>> all_same([1, 1, 2, 1])
        False
    """
    if not items:
        return True
    first = items[0]
    return all(item == first for item in items)


# ==================== String Formatting ====================

def format_number(value: float, precision: int = 4, scientific: bool = False) -> str:
    """
    Format a number for display.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        scientific: If True, use scientific notation for very large/small numbers
    
    Returns:
        Formatted string
    
    Example:
        >>> format_number(3.14159, precision=2)
        '3.14'
        >>> format_number(0.000001, scientific=True)
        '1.0000e-06'
    """
    if scientific and (abs(value) < 1e-4 or abs(value) > 1e6):
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def format_matrix_string(data: List[List[float]], precision: int = 4) -> str:
    """
    Format a matrix as a nicely aligned string.
    
    Args:
        data: Matrix data
        precision: Number of decimal places
    
    Returns:
        Formatted string representation
    
    Example:
        >>> data = [[1, 2], [3, 4]]
        >>> print(format_matrix_string(data))
        [1.0000  2.0000]
        [3.0000  4.0000]
    """
    if not data:
        return "[]"
    
    # Format all numbers
    formatted = [[format_number(val, precision) for val in row] for row in data]
    
    # Find max width per column
    cols = len(data[0])
    widths = [0] * cols
    for row in formatted:
        for j, val in enumerate(row):
            widths[j] = max(widths[j], len(val))
    
    # Build string
    lines = []
    for row in formatted:
        padded = [val.rjust(widths[j]) for j, val in enumerate(row)]
        lines.append("[" + "  ".join(padded) + "]")
    
    return "\n".join(lines)


def format_vector_string(components: Tuple[float, ...], precision: int = 4) -> str:
    """
    Format a vector as a string.
    
    Args:
        components: Vector components
        precision: Number of decimal places
    
    Returns:
        Formatted string representation
    
    Example:
        >>> format_vector_string((1, 2, 3), precision=2)
        '(1.00, 2.00, 3.00)'
    """
    formatted = [format_number(c, precision) for c in components]
    return "(" + ", ".join(formatted) + ")"


# ==================== Range and Sequence Utilities ====================

def linspace(start: float, stop: float, num: int = 50) -> List[float]:
    """
    Generate evenly spaced numbers over a specified interval.
    
    Args:
        start: Start value
        stop: End value
        num: Number of points to generate
    
    Returns:
        List of evenly spaced values
    
    Example:
        >>> linspace(0, 10, 5)
        [0.0, 2.5, 5.0, 7.5, 10.0]
    """
    if num <= 0:
        return []
    if num == 1:
        return [start]
    
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def arange(start: float, stop: float, step: float = 1.0) -> List[float]:
    """
    Generate values in a range with a given step.
    
    Args:
        start: Start value (inclusive)
        stop: End value (exclusive)
        step: Step size
    
    Returns:
        List of values
    
    Example:
        >>> arange(0, 5, 1)
        [0.0, 1.0, 2.0, 3.0, 4.0]
        >>> arange(0, 1, 0.25)
        [0.0, 0.25, 0.5, 0.75]
    """
    if step == 0:
        raise ValueError("Step cannot be zero")
    
    result = []
    current = start
    if step > 0:
        while current < stop:
            result.append(current)
            current += step
    else:
        while current > stop:
            result.append(current)
            current += step
    
    return result


# ==================== Comparison Utilities ====================

def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-10) -> bool:
    """
    Check if two floats are close within tolerance.
    
    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
    
    Returns:
        True if values are close
    
    Example:
        >>> is_close(1.0, 1.0000000001)
        True
        >>> is_close(1.0, 1.1)
        False
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def all_close(list1: List[float], list2: List[float], 
              rel_tol: float = 1e-9, abs_tol: float = 1e-10) -> bool:
    """
    Check if all elements in two lists are close.
    
    Args:
        list1: First list
        list2: Second list
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
    
    Returns:
        True if all corresponding elements are close
    
    Example:
        >>> all_close([1.0, 2.0, 3.0], [1.0000001, 2.0000001, 3.0000001])
        True
    """
    if len(list1) != len(list2):
        return False
    return all(is_close(a, b, rel_tol, abs_tol) for a, b in zip(list1, list2))


# ==================== Function Utilities ====================

def compose(*functions: Callable) -> Callable:
    """
    Compose multiple functions into a single function.
    
    Args:
        *functions: Functions to compose (applied right to left)
    
    Returns:
        Composed function
    
    Example:
        >>> double = lambda x: x * 2
        >>> add_one = lambda x: x + 1
        >>> f = compose(double, add_one)
        >>> f(5)  # double(add_one(5)) = double(6) = 12
        12
    """
    def composed(x):
        for func in reversed(functions):
            x = func(x)
        return x
    return composed


def memoize(func: Callable) -> Callable:
    """
    Decorator to cache function results.
    
    Args:
        func: Function to memoize
    
    Returns:
        Memoized function
    
    Example:
        >>> @memoize
        >>> def fib(n):
        >>>     if n <= 1: return n
        >>>     return fib(n-1) + fib(n-2)
    """
    cache = {}
    
    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return memoized


# ==================== Statistics Utilities ====================

def mean(values: List[float]) -> float:
    """
    Calculate arithmetic mean.
    
    Args:
        values: List of numbers
    
    Returns:
        Mean value
    
    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def variance(values: List[float], ddof: int = 0) -> float:
    """
    Calculate variance.
    
    Args:
        values: List of numbers
        ddof: Delta degrees of freedom (0 for population, 1 for sample)
    
    Returns:
        Variance
    
    Example:
        >>> variance([1, 2, 3, 4, 5])
        2.0
    """
    if not values:
        raise ValueError("Cannot compute variance of empty list")
    if len(values) - ddof <= 0:
        raise ValueError("Not enough data points")
    
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - ddof)


def std_dev(values: List[float], ddof: int = 0) -> float:
    """
    Calculate standard deviation.
    
    Args:
        values: List of numbers
        ddof: Delta degrees of freedom (0 for population, 1 for sample)
    
    Returns:
        Standard deviation
    
    Example:
        >>> std_dev([1, 2, 3, 4, 5])
        1.4142135623730951
    """
    return math.sqrt(variance(values, ddof))


# ==================== Type Conversion Utilities ====================

def ensure_float_list(values: Union[List[float], Tuple[float, ...]]) -> List[float]:
    """
    Ensure a sequence is a list of floats.
    
    Args:
        values: Sequence of numbers
    
    Returns:
        List of floats
    
    Example:
        >>> ensure_float_list([1, 2, 3])
        [1.0, 2.0, 3.0]
        >>> ensure_float_list((1, 2, 3))
        [1.0, 2.0, 3.0]
    """
    return [float(v) for v in values]


def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
    
    Returns:
        Result of division or default
    
    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0, default=0)
        0.0
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator
