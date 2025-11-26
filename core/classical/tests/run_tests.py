"""
Run all classical module tests.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def run_all_tests():
    """Run all test modules."""
    print("=" * 60)
    print("Running Classical Module Tests")
    print("=" * 60)
    print()
    
    # Import and run each test module
    modules = [
        ("LivniumCoreSystem", "test_livnium_core_system"),
        ("DataCube", "test_datacube"),
        ("DataGrid", "test_datagrid"),
    ]
    
    results = {}
    
    for name, module_name in modules:
        print(f"\n{'=' * 60}")
        print(f"Testing {name}")
        print('=' * 60)
        
        try:
            module = __import__(f"core.classical.tests.{module_name}", fromlist=[module_name])
            
            # Run the module's main block if it exists
            if hasattr(module, '__name__') and module.__name__ == f"core.classical.tests.{module_name}":
                # Execute the main block
                if '__main__' in dir(module):
                    # The module will run its own tests when imported with __main__
                    pass
            
            # Try to run the module directly
            import subprocess
            result = subprocess.run(
                [sys.executable, f"{Path(__file__).parent}/{module_name}.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(result.stdout)
                results[name] = "PASSED"
            else:
                print(result.stdout)
                print(result.stderr)
                results[name] = "FAILED"
                
        except Exception as e:
            print(f"Error running {name} tests: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "ERROR"
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, status in results.items():
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {name}: {status}")
    
    all_passed = all(status == "PASSED" for status in results.values())
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

