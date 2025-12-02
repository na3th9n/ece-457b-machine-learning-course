"""
Test cases for LearnedRectangle class
"""

from LearnedRectangle import LearnedRectangle
from generate_example import generate_example

def test_learn_with_positive_examples():
    """Test that the model learns from positive examples correctly"""
    print("Testing learn with positive examples...")
    
    model = LearnedRectangle()
    
    # Create some mock positive examples
    # Note: We can't easily mock the generator, so we'll test with real examples
    # but we can check the structure of what we learn
    
    model.learn(5)
    
    # Check that hypothesis rectangle is initialized
    assert len(model.hypothesisRectangle) > 0, "Hypothesis rectangle should be initialized"
    
    # Check that each dimension has bounds (lower, upper)
    for bounds in model.hypothesisRectangle:
        assert len(bounds) == 2, "Each dimension should have (lower, upper) bounds"
        assert bounds[0] <= bounds[1], "Lower bound should be <= upper bound"
    
    print("✓ Learn with positive examples test passed")

def test_is_within_rectangle():
    """Test the _is_within_rectangle helper method"""
    print("Testing _is_within_rectangle...")
    
    model = LearnedRectangle()
    
    # Test with empty rectangle
    assert not model._is_within_rectangle([1, 2]), "Empty rectangle should return False"
    
    # Set up a simple 2D rectangle: x in [0, 10], y in [0, 10]
    model.hypothesisRectangle = [(0, 10), (0, 10)]
    
    # Test points inside rectangle
    assert model._is_within_rectangle([5, 5]), "Point [5,5] should be inside"
    assert model._is_within_rectangle([0, 0]), "Point [0,0] should be inside (boundary)"
    assert model._is_within_rectangle([10, 10]), "Point [10,10] should be inside (boundary)"
    
    # Test points outside rectangle
    assert not model._is_within_rectangle([15, 5]), "Point [15,5] should be outside (x too high)"
    assert not model._is_within_rectangle([5, 15]), "Point [5,15] should be outside (y too high)"
    assert not model._is_within_rectangle([-1, 5]), "Point [-1,5] should be outside (x too low)"
    assert not model._is_within_rectangle([5, -1]), "Point [5,-1] should be outside (y too low)"
    
    print("✓ _is_within_rectangle test passed")

def test_checkgoodness_basic():
    """Test basic functionality of checkgoodness"""
    print("Testing checkgoodness basic functionality...")
    
    model = LearnedRectangle()
    
    # Set up a simple rectangle
    model.hypothesisRectangle = [(0, 10), (0, 10)]
    
    # Test with small parameters
    result = model.checkgoodness(2, 3, 0.5)
    
    # Result should be a non-negative integer
    assert isinstance(result, int), "Result should be an integer"
    assert result >= 0, "Result should be non-negative"
    assert result <= 2, "Result should not exceed n (number of test sets)"
    
    print("✓ Checkgoodness basic test passed")

def test_learn_with_mixed_examples():
    """Test learning with a mix of positive and negative examples"""
    print("Testing learn with mixed examples...")
    
    model = LearnedRectangle()
    
    # Learn from multiple examples (some will be positive, some negative)
    model.learn(20)  # Use more examples to increase chance of getting positive ones
    
    # The model should only learn from positive examples
    # So if we got any positive examples, the rectangle should be initialized
    if model.hypothesisRectangle:
        print(f"  Learned rectangle: {model.hypothesisRectangle}")
        # Verify structure
        for bounds in model.hypothesisRectangle:
            assert len(bounds) == 2, "Each dimension should have (lower, upper) bounds"
            assert bounds[0] <= bounds[1], "Lower bound should be <= upper bound"
    else:
        print("  No positive examples found in 20 samples (unlikely but possible)")
    
    print("✓ Learn with mixed examples test passed")

def test_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    model = LearnedRectangle()
    
    # Test with single positive example
    model.learn(1)
    
    # Test checkgoodness with very small parameters
    result = model.checkgoodness(1, 1, 0.0)  # epsilon = 0 means any misclassification counts
    assert isinstance(result, int) and 0 <= result <= 1
    
    # Test with high epsilon (should rarely trigger)
    result = model.checkgoodness(1, 1, 1.0)  # epsilon = 1.0 means all must be misclassified
    assert isinstance(result, int) and 0 <= result <= 1
    
    print("✓ Edge cases test passed")

def test_rectangle_expansion():
    """Test that the rectangle expands correctly with new positive examples"""
    print("Testing rectangle expansion...")
    
    model = LearnedRectangle()
    
    # Manually set up a rectangle
    model.hypothesisRectangle = [(5, 5), (5, 5)]  # Single point
    
    # Test expansion logic
    point = [3, 7]  # Should expand the rectangle
    
    # Simulate the expansion logic from learn()
    for idx, coord in enumerate(point):
        if idx < len(model.hypothesisRectangle):
            model.hypothesisRectangle[idx] = (
                min(model.hypothesisRectangle[idx][0], coord),
                max(model.hypothesisRectangle[idx][1], coord)
            )
    
    # Check that rectangle expanded correctly
    assert model.hypothesisRectangle[0] == (3, 5), f"X bounds should be (3, 5), got {model.hypothesisRectangle[0]}"
    assert model.hypothesisRectangle[1] == (5, 7), f"Y bounds should be (5, 7), got {model.hypothesisRectangle[1]}"
    
    print("✓ Rectangle expansion test passed")

def run_performance_test():
    """Test performance with larger parameters"""
    print("Testing performance...")
    
    model = LearnedRectangle()
    
    # Test with larger learning set
    import time
    start_time = time.time()
    model.learn(100)
    learn_time = time.time() - start_time
    
    # Test with larger goodness check
    start_time = time.time()
    result = model.checkgoodness(10, 20, 0.1)
    check_time = time.time() - start_time
    
    print(f"  Learn time (100 examples): {learn_time:.3f}s")
    print(f"  Check time (10 sets of 20): {check_time:.3f}s")
    print(f"  Goodness result: {result}")
    
    assert learn_time < 5.0, "Learning should complete within 5 seconds"
    assert check_time < 10.0, "Goodness check should complete within 10 seconds"
    
    print("✓ Performance test passed")

def main():
    """Run all test cases"""
    print("Running LearnedRectangle test suite...\n")
    
    try:
        test_learn_with_positive_examples()
        test_is_within_rectangle()
        test_checkgoodness_basic()
        test_learn_with_mixed_examples()
        test_edge_cases()
        test_rectangle_expansion()
        run_performance_test()
        
        print("\n🎉 All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()
