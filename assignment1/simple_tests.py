"""
Simple test cases for LearnedRectangle - easier to debug
"""

from LearnedRectangle import LearnedRectangle

def test_empty_rectangle():
    """Test behavior with empty rectangle"""
    print("Testing empty rectangle...")
    
    model = LearnedRectangle()
    
    # Test _is_within_rectangle with empty rectangle
    assert not model._is_within_rectangle([1, 2, 3]), "Empty rectangle should return False"
    assert not model._is_within_rectangle([]), "Empty rectangle should return False for empty point"
    
    print("✓ Empty rectangle test passed")

def test_manual_rectangle():
    """Test with manually set rectangle"""
    print("Testing manual rectangle...")
    
    model = LearnedRectangle()
    
    # Set up a 2D rectangle: x in [0, 10], y in [0, 10]
    model.hypothesisRectangle = [(0, 10), (0, 10)]
    
    # Test various points
    test_cases = [
        ([5, 5], True, "Center point"),
        ([0, 0], True, "Corner point (lower-left)"),
        ([10, 10], True, "Corner point (upper-right)"),
        ([0, 10], True, "Corner point (upper-left)"),
        ([10, 0], True, "Corner point (lower-right)"),
        ([15, 5], False, "Outside (x too high)"),
        ([5, 15], False, "Outside (y too high)"),
        ([-1, 5], False, "Outside (x too low)"),
        ([5, -1], False, "Outside (y too low)"),
        ([15, 15], False, "Outside (both too high)"),
        ([-1, -1], False, "Outside (both too low)"),
    ]
    
    for point, expected, description in test_cases:
        result = model._is_within_rectangle(point)
        assert result == expected, f"{description}: point {point} should be {expected}, got {result}"
        print(f"  ✓ {description}: {point} -> {result}")
    
    print("✓ Manual rectangle test passed")

def test_3d_rectangle():
    """Test with 3D rectangle"""
    print("Testing 3D rectangle...")
    
    model = LearnedRectangle()
    
    # Set up a 3D rectangle: x in [0, 5], y in [1, 6], z in [2, 7]
    model.hypothesisRectangle = [(0, 5), (1, 6), (2, 7)]
    
    # Test various 3D points
    test_cases = [
        ([2, 3, 4], True, "Center point"),
        ([0, 1, 2], True, "Corner point (min)"),
        ([5, 6, 7], True, "Corner point (max)"),
        ([6, 3, 4], False, "Outside (x too high)"),
        ([2, 7, 4], False, "Outside (y too high)"),
        ([2, 3, 8], False, "Outside (z too high)"),
        ([-1, 3, 4], False, "Outside (x too low)"),
        ([2, 0, 4], False, "Outside (y too low)"),
        ([2, 3, 1], False, "Outside (z too low)"),
    ]
    
    for point, expected, description in test_cases:
        result = model._is_within_rectangle(point)
        assert result == expected, f"{description}: point {point} should be {expected}, got {result}"
        print(f"  ✓ {description}: {point} -> {result}")
    
    print("✓ 3D rectangle test passed")

def test_rectangle_expansion():
    """Test rectangle expansion logic"""
    print("Testing rectangle expansion...")
    
    model = LearnedRectangle()
    
    # Start with a single point
    model.hypothesisRectangle = [(5, 5), (5, 5)]
    
    # Test expanding with a point that extends both dimensions
    new_point = [3, 7]
    
    # Simulate the expansion logic
    for idx, coord in enumerate(new_point):
        if idx < len(model.hypothesisRectangle):
            model.hypothesisRectangle[idx] = (
                min(model.hypothesisRectangle[idx][0], coord),
                max(model.hypothesisRectangle[idx][1], coord)
            )
    
    # Check results
    assert model.hypothesisRectangle[0] == (3, 5), f"X should expand to (3, 5), got {model.hypothesisRectangle[0]}"
    assert model.hypothesisRectangle[1] == (5, 7), f"Y should expand to (5, 7), got {model.hypothesisRectangle[1]}"
    
    print("  ✓ Rectangle expanded correctly")
    
    # Test that the new rectangle works
    assert model._is_within_rectangle([3, 5]), "Original point should still be inside"
    assert model._is_within_rectangle([5, 7]), "New point should be inside"
    assert model._is_within_rectangle([4, 6]), "Point in middle should be inside"
    assert not model._is_within_rectangle([2, 6]), "Point outside should be outside"
    assert not model._is_within_rectangle([6, 6]), "Point outside should be outside"
    
    print("✓ Rectangle expansion test passed")

def test_checkgoodness_with_known_rectangle():
    """Test checkgoodness with a known rectangle"""
    print("Testing checkgoodness with known rectangle...")
    
    model = LearnedRectangle()
    
    # Set up a rectangle that should be close to the target
    # Based on generate_example.py, target is x in [10, 1000], y in [10, 1000]
    model.hypothesisRectangle = [(10, 1000), (10, 1000)]
    
    # Test with small parameters
    result = model.checkgoodness(2, 5, 0.2)
    
    # Result should be reasonable
    assert isinstance(result, int), "Result should be integer"
    assert 0 <= result <= 2, "Result should be between 0 and n"
    
    print(f"  ✓ Checkgoodness result: {result}")
    print("✓ Checkgoodness test passed")

def main():
    """Run simple tests"""
    print("Running simple LearnedRectangle tests...\n")
    
    try:
        test_empty_rectangle()
        print()
        test_manual_rectangle()
        print()
        test_3d_rectangle()
        print()
        test_rectangle_expansion()
        print()
        test_checkgoodness_with_known_rectangle()
        
        print("\n🎉 All simple tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
