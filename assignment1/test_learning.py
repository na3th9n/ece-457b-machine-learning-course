"""
Test the learning algorithm specifically
"""

from LearnedRectangle import LearnedRectangle
from generate_example import generate_example

def test_learning_algorithm():
    """Test the learning algorithm step by step"""
    print("Testing learning algorithm...")
    
    model = LearnedRectangle()
    
    # Get some examples to see what we're working with
    print("Getting sample data...")
    samples = []
    for i in range(20):
        sample = next(generate_example())
        samples.append(sample)
        print(f"  Sample {i+1}: {sample}")
    
    # Count positive vs negative examples
    positive_count = sum(1 for sample in samples if sample[1])
    negative_count = len(samples) - positive_count
    
    print(f"\nFound {positive_count} positive examples and {negative_count} negative examples")
    
    # Test learning with these examples
    print("\nTesting learning...")
    model.learn(20)
    
    if model.hypothesisRectangle:
        print(f"Learned rectangle: {model.hypothesisRectangle}")
        
        # Test the learned rectangle against the original samples
        print("\nTesting learned rectangle against original samples...")
        correct_predictions = 0
        total_predictions = 0
        
        for i, sample in enumerate(samples):
            point = sample[0]
            actual_label = sample[1]
            predicted_label = model._is_within_rectangle(point)
            
            is_correct = (predicted_label == actual_label)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  Sample {i+1}: {point} -> predicted: {predicted_label}, actual: {actual_label} {status}")
        
        accuracy = correct_predictions / total_predictions
        print(f"\nAccuracy on training data: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        
    else:
        print("No positive examples found - rectangle not initialized")
    
    print("✓ Learning algorithm test completed")

def test_learning_with_different_sample_sizes():
    """Test learning with different sample sizes"""
    print("\nTesting learning with different sample sizes...")
    
    sample_sizes = [5, 10, 20, 50]
    
    for m in sample_sizes:
        print(f"\nTesting with m={m} samples...")
        model = LearnedRectangle()
        model.learn(m)
        
        if model.hypothesisRectangle:
            print(f"  Learned rectangle: {model.hypothesisRectangle}")
            
            # Test on some new examples
            test_samples = [next(generate_example()) for _ in range(10)]
            correct = 0
            for sample in test_samples:
                point = sample[0]
                actual = sample[1]
                predicted = model._is_within_rectangle(point)
                if predicted == actual:
                    correct += 1
            
            accuracy = correct / len(test_samples)
            print(f"  Test accuracy: {accuracy:.2%}")
        else:
            print("  No positive examples found")

def test_goodness_check():
    """Test the goodness check function"""
    print("\nTesting goodness check...")
    
    model = LearnedRectangle()
    model.learn(20)
    
    if model.hypothesisRectangle:
        print(f"Testing with learned rectangle: {model.hypothesisRectangle}")
        
        # Test with different parameters
        test_cases = [
            (2, 5, 0.2),
            (3, 10, 0.1),
            (1, 20, 0.05),
        ]
        
        for n, k, epsilon in test_cases:
            result = model.checkgoodness(n, k, epsilon)
            print(f"  n={n}, k={k}, epsilon={epsilon} -> result: {result}")
            assert 0 <= result <= n, f"Result should be between 0 and {n}"
    else:
        print("No rectangle learned - skipping goodness check")

def main():
    """Run learning tests"""
    print("Running learning algorithm tests...\n")
    
    try:
        test_learning_algorithm()
        test_learning_with_different_sample_sizes()
        test_goodness_check()
        
        print("\n🎉 All learning tests completed!")
        
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
