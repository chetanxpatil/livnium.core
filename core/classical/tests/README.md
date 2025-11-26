# Classical Module Tests

Assertion-based tests for the classical Livnium core system.

## Test Files

- **`test_livnium_core_system.py`**: Tests for `LivniumCoreSystem`
  - Basic initialization
  - Cell classification (core, center, edge, corner)
  - Symbolic weight formula (SW = 9·f)
  - Total symbolic weight calculation
  - Class count invariants
  - Symbol alphabet
  - Rotation preservation of invariants
  - Rotation group operations
  - Semantic polarity
  - Local observers
  - Generalized N (3, 5, 7, ...)
  - Feature switches
  - System summary

- **`test_datacube.py`**: Tests for `DataCube` (even-dimensional resource grids)
  - Basic initialization
  - Invalid size validation
  - No center cell (even N)
  - Data storage and retrieval
  - Invalid coordinates
  - Clear operation
  - Numpy conversion (to/from)
  - Numpy roundtrip
  - Get all data

- **`test_datagrid.py`**: Tests for `DataGrid` (2D resource grids)
  - Basic initialization
  - Invalid size validation
  - Coordinate ranges (odd/even N)
  - Data storage and retrieval
  - Invalid coordinates
  - Clear operation
  - Numpy conversion (to/from)
  - Numpy roundtrip
  - Get all data
  - Various grid sizes

## Running Tests

Run individual test files:
```bash
python3 core/classical/tests/test_livnium_core_system.py
python3 core/classical/tests/test_datacube.py
python3 core/classical/tests/test_datagrid.py
```

Or use the test runner:
```bash
python3 core/classical/tests/run_tests.py
```

## Test Style

All tests use Python `assert` statements. Each test function:
- Has a descriptive name starting with `test_`
- Uses clear assertions with helpful error messages
- Tests one specific aspect of functionality
- Can be run independently

## Test Coverage

The tests cover:
- ✅ Core system initialization and configuration
- ✅ Cell classification and symbolic weight
- ✅ Rotation operations and invariants
- ✅ Observer system (global and local)
- ✅ Symbol alphabet
- ✅ Data storage (DataCube and DataGrid)
- ✅ Numpy integration
- ✅ Error handling and validation
- ✅ Generalized N support (3, 5, 7, ...)

