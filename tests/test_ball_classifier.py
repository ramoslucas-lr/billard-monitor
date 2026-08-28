import pytest
import numpy as np
import cv2 as cv
from billard.ball_classifier import (
    Circle,
    find_nearest,
    z3_operation,
    unsharp_mask,
    cut_circle,
    get_circle_descriptors,
)

def test_circle_initialization():
    """Test the initialization of the Circle class."""
    circle = Circle(10, 20, 15)
    assert circle.x == 10
    assert circle.y == 20
    assert circle.r == 15

def test_find_nearest():
    """Test find_nearest function for various scenarios."""
    array = [10, 20, 30, 40, 50]
    # Exact match
    assert find_nearest(array, 30) == 30
    # Rounds up
    assert find_nearest(array, 36) == 40
    # Rounds down
    assert find_nearest(array, 24) == 20
    # Handles negative values (if applicable in general, although context is image coordinates)
    array_with_negatives = [-10, 0, 10]
    assert find_nearest(array_with_negatives, -8) == -10

def test_z3_operation():
    """Test z3_operation function for all morphological operations."""
    # Create a dummy image
    img = np.ones((20, 20), dtype=np.uint8) * 128

    for op in range(1, 6):
        result = z3_operation(op, img)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    # Test fallback
    result_fallback = z3_operation(99, img)
    assert np.array_equal(result_fallback, img)

def test_unsharp_mask():
    """Test unsharp_mask function."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    result = unsharp_mask(img)

    assert result.shape == img.shape
    assert result.dtype == img.dtype

def test_cut_circle():
    """Test cut_circle function and edge cases (padding)."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 200

    # Normal case: circle in the middle
    circle = Circle(50, 50, 10) # radius passed here doesn't matter for cut_circle, hardcoded r=10 inside
    result = cut_circle(circle, frame)
    assert result.shape == (20, 20, 3) # 2*r, 2*r, 3

    # Edge case: circle near the top-left border (requires padding)
    circle_tl = Circle(5, 5, 10)
    result_tl = cut_circle(circle_tl, frame)
    assert result_tl.shape == (20, 20, 3)

    # Edge case: circle near the bottom-right border (requires padding)
    circle_br = Circle(95, 95, 10)
    result_br = cut_circle(circle_br, frame)
    assert result_br.shape == (20, 20, 3)

def test_get_circle_descriptors():
    """Test get_circle_descriptors function."""
    # Create a dummy cut circle (20x20x3)
    final = np.ones((20, 20, 3), dtype=np.uint8) * 150
    # Make it slightly colorful to ensure non-zero hsv values
    final[:, :, 0] = 100 # B
    final[:, :, 1] = 150 # G
    final[:, :, 2] = 200 # R

    result = get_circle_descriptors(final)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 6)

    # Checking that the output values are not all zero (to ensure cv.mean did something)
    assert np.any(result > 0)
