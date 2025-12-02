"""
Test runner for LearnedRectangle
"""

import sys
import os

def run_simple_tests():
    """Run the simple unit tests"""
    print("=" * 50)
    print("RUNNING SIMPLE TESTS")
    print("=" * 50)
    
    try:
        from simple_tests import main as simple_main
        simple_main()
        return True
    except Exception as e:
        print(f"Simple tests failed: {e}")
        return False

def run_learning_tests():
    """Run the learning algorithm tests"""
    print("\n" + "=" * 50)
    print("RUNNING LEARNING TESTS")
    print("=" * 50)
    
    try:
        from test_learning import main as learning_main
        learning_main()
        return True
    except Exception as e:
        print(f"Learning tests failed: {e}")
        return False

def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    print("\n" + "=" * 50)
    print("RUNNING COMPREHENSIVE TESTS")
    print("=" * 50)
    
    try:
        from test_learned_rectangle import main as comprehensive_main
        comprehensive_main()
        return True
    except Exception as e:
        print(f"Comprehensive tests failed: {e}")
        return False

def main():
    """Run all test suites"""
    print("LearnedRectangle Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("LearnedRectangle.py"):
        print("Error: LearnedRectangle.py not found. Make sure you're in the right directory.")
        return
    
    # Run tests in order of complexity
    test_results = []
    
    # 1. Simple tests (fastest, most reliable)
    test_results.append(("Simple Tests", run_simple_tests()))
    
    # 2. Learning tests (tests actual algorithm)
    test_results.append(("Learning Tests", run_learning_tests()))
    
    # 3. Comprehensive tests (full integration)
    test_results.append(("Comprehensive Tests", run_comprehensive_tests()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All test suites passed!")
    else:
        print("\n❌ Some test suites failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
