"""
Simple import test for recursive module.

Verifies that all modules can be imported and basic structure is correct.
"""

import sys
from pathlib import Path
import ast

def test_file_syntax(file_path):
    """Test that a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def test_imports():
    """Test that recursive module files exist and have valid syntax."""
    core_t_path = Path(__file__).parent.parent
    recursive_dir = core_t_path / "recursive"
    
    files_to_test = [
        "recursive_simplex_engine.py",
        "simplex_subdivision.py",
        "recursive_projection.py",
        "recursive_conservation.py",
        "moksha_engine.py",
        "__init__.py",
    ]
    
    print("Testing recursive module files...")
    all_passed = True
    
    for filename in files_to_test:
        file_path = recursive_dir / filename
        if not file_path.exists():
            print(f"❌ {filename} does not exist")
            all_passed = False
            continue
        
        is_valid, error = test_file_syntax(file_path)
        if is_valid:
            print(f"✓ {filename} - syntax valid")
        else:
            print(f"❌ {filename} - syntax error: {error}")
            all_passed = False
    
    # Check README exists
    readme_path = recursive_dir / "README.md"
    if readme_path.exists():
        print("✓ README.md exists")
    else:
        print("❌ README.md does not exist")
        all_passed = False
    
    return all_passed

def test_structure():
    """Test that the structure matches core/recursive."""
    core_t_path = Path(__file__).parent.parent
    core_recursive = core_t_path.parent / "core" / "recursive"
    t_recursive = core_t_path / "recursive"
    
    print("\nComparing structure with core/recursive...")
    
    # Files that should exist in both
    expected_files = [
        "__init__.py",
        "moksha_engine.py",
        "README.md",
    ]
    
    # Core-specific files
    core_files = {
        "recursive_geometry_engine.py": "recursive_simplex_engine.py",
        "geometry_subdivision.py": "simplex_subdivision.py",
    }
    
    all_match = True
    
    for filename in expected_files:
        core_file = core_recursive / filename
        t_file = t_recursive / filename
        
        if core_file.exists() and not t_file.exists():
            print(f"❌ Missing: {filename}")
            all_match = False
        elif core_file.exists() and t_file.exists():
            print(f"✓ {filename} exists in both")
    
    for core_name, t_name in core_files.items():
        core_file = core_recursive / core_name
        t_file = t_recursive / t_name
        
        if core_file.exists() and not t_file.exists():
            print(f"❌ Missing equivalent: {t_name} (for {core_name})")
            all_match = False
        elif core_file.exists() and t_file.exists():
            print(f"✓ {t_name} exists (equivalent to {core_name})")
    
    return all_match

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Recursive Module for core-t")
    print("=" * 60)
    
    syntax_ok = test_imports()
    structure_ok = test_structure()
    
    print("\n" + "=" * 60)
    if syntax_ok and structure_ok:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)

