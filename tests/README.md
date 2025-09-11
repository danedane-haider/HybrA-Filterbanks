# HybrA-Filterbanks Test Suite

This directory contains a comprehensive test suite for the HybrA-Filterbanks package, a PyTorch library providing state-of-the-art auditory-inspired filterbanks for audio processing and deep learning applications.

## Test Structure

The test suite is organized into several modules:

### Core Test Files

- **`test_basic.py`** - Essential functionality tests that work with the current API
  - Basic functionality for all filterbank classes (ISAC, HybrA, ISACSpec, ISACCC)
  - Forward/backward passes, gradient computation, state dict handling
  - Integration with PyTorch neural networks
  - Edge cases and error handling

- **`test_isac.py`** - Comprehensive tests for ISAC filterbank
- **`test_hybra.py`** - Comprehensive tests for HybrA filterbank  
- **`test_isac_mel.py`** - Tests for ISACSpec spectrogram functionality
- **`test_isac_mfcc.py`** - Tests for ISACCC cepstral coefficient extraction
- **`test_utils.py`** - Tests for utility functions (frame bounds, circular convolution, etc.)
- **`test_integration.py`** - Integration and end-to-end workflow tests

### Configuration Files

- **`conftest.py`** - Shared fixtures and test configuration
- **`__init__.py`** - Empty init file for test package

## Running Tests

The test suite uses `pytest` and can be run using `uv`:

### Basic Test Run
```bash
uv run pytest tests/test_basic.py -v
```

### Run All Tests
```bash
uv run pytest tests/ -v
```

### Run with Coverage
```bash
uv run pytest tests/test_basic.py --cov=hybra --cov-report=term-missing
```

### Run Specific Test Categories
```bash
# Run only basic functionality tests
uv run pytest tests/test_basic.py::TestBasicFunctionality -v

# Run only integration tests  
uv run pytest tests/test_basic.py::TestIntegrationBasic -v

# Run only utility function tests
uv run pytest tests/test_basic.py::TestUtilityFunctions -v
```

## Test Coverage

Current test coverage (with `test_basic.py`):
- **Total Coverage**: 61%
- **Core Modules**:
  - `hybra/__init__.py`: 100%
  - `hybra/isac_mfcc.py`: 81%
  - `hybra/isac.py`: 75%
  - `hybra/isac_mel.py`: 70%
  - `hybra/utils.py`: 70%
  - `hybra/hybridfilterbank.py`: 68%

## Key Test Areas

### 1. **Basic Functionality**
- Filterbank initialization with various parameters
- Forward pass shape validation
- Batch processing support
- Different auditory scales (mel, erb, log10)

### 2. **Machine Learning Integration**
- Gradient flow through learnable parameters
- State dict save/load functionality
- Integration with PyTorch neural networks
- Training stability

### 3. **Mathematical Properties**
- Perfect reconstruction (within tolerance)
- Frame bounds computation
- Condition number analysis
- Circular convolution properties

### 4. **Edge Cases**
- Zero input signals
- Different signal lengths
- Extreme parameter values
- Error handling and validation

### 5. **End-to-End Workflows**
- Complete audio processing pipelines
- Feature extraction chains (ISAC → ISACSpec → ISACCC)
- Neural network integration patterns

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

- `sample_audio` - 1-second audio signal with multiple frequency components
- `short_audio` - Short 1024-sample audio signal for quick tests
- `noise_audio` - White noise signal for robustness testing
- `test_parameters` - Standard filterbank parameters
- `small_test_parameters` - Smaller parameters for faster testing
- `tolerance` / `loose_tolerance` - Numerical comparison tolerances

## API Compatibility Notes

The test suite has been designed to work with the actual HybrA-Filterbanks API:

- **Complex Outputs**: Filterbanks produce complex-valued outputs, requiring `.abs()` conversion for gradient computation
- **Dynamic Shapes**: Output shapes depend on internal processing and may not exactly match input lengths
- **Tensor Attributes**: Some attributes (like `kernel_size`) are tensors rather than scalars

## Dependencies

Test dependencies are managed via `uv` and defined in `pyproject.toml`:

```toml
[dependency-groups.test]
pytest>=8.4.2
pytest-cov>=7.0.0
pytest-mock>=3.15.0
```

## Continuous Integration

The test suite is designed to be robust and should work in CI environments:

- All tests use reproducible random seeds
- Visualization tests are mocked to avoid display requirements  
- Tests handle missing optional dependencies gracefully
- Platform-specific tests are properly skipped

## Contributing

When adding new tests:

1. Follow the existing naming conventions (`test_*.py`)
2. Use appropriate fixtures from `conftest.py`
3. Test both successful cases and error conditions
4. Ensure tests work with complex-valued outputs
5. Add integration tests for new functionality
6. Verify tests pass in isolation and as part of the full suite

## Future Improvements

Potential areas for test suite enhancement:

1. **Performance Tests**: Add benchmarks for different configurations
2. **Property-Based Testing**: Use hypothesis for more comprehensive testing
3. **Audio-Specific Tests**: Add tests with real audio samples
4. **Visualization Tests**: More comprehensive testing of plotting functions
5. **Device Tests**: Add GPU/MPS device compatibility tests
6. **Numerical Stability**: More extensive testing of edge cases and ill-conditioned systems