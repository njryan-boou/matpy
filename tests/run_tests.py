"""
Test runner script for matpy test suite.

Run all tests with:
    python tests/run_tests.py

Run specific test file:
    python tests/run_tests.py test_vector_core
    python tests/run_tests.py test_vector_ops

Run with verbose output:
    python tests/run_tests.py -v
"""

import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_all_tests():
    """Discover and run all tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_name):
    """Run a specific test module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] not in ['-v', '--verbose']:
        # Run specific test
        test_name = sys.argv[1]
        if not test_name.startswith('test_'):
            test_name = f'test_{test_name}'
        success = run_specific_test(test_name)
    else:
        # Run all tests
        print("Running all matpy tests...\n")
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
