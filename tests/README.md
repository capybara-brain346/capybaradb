# CapybaraDB Test Suite

This directory contains comprehensive unit and integration tests for the CapybaraDB library.

## Test Structure

- `test_logger.py` - Tests for the logger module
- `test_utils.py` - Tests for file extraction utilities and OCR processing
- `test_model.py` - Tests for the EmbeddingModel class
- `test_main.py` - Tests for the CapybaraDB main class
- `test_integration.py` - Integration tests using real data files
- `conftest.py` - Pytest fixtures and configuration
- `test_runner.py` - Test runner script

## Running Tests

### Install test dependencies
```bash
pip install -e ".[test]"
```

### Run all tests
```bash
pytest
```

### Run specific test modules
```bash
pytest tests/test_utils.py
pytest tests/test_model.py
pytest tests/test_main.py
```

### Run with coverage
```bash
pytest --cov=capybaradb --cov-report=html
```

### Run integration tests only
```bash
pytest -m integration
```

### Run unit tests only
```bash
pytest -m unit
```

## Test Data

The tests use real data files from the `data/` directory:
- `CNN-Based Classifiers and Fine-Tune.txt`
- `Fine-Tuned LLM_SLM Use Cases in TrackML-Backend.pdf`
- `research-paper-format.docx`

## Mocking Strategy

The tests use extensive mocking to avoid:
- Downloading large ML models during testing
- GPU dependencies
- External file system dependencies
- Network calls

Key mocked components:
- `transformers` models and tokenizers
- `torch` CUDA availability
- File I/O operations
- OCR processing

## Test Coverage

The test suite covers:
- ✅ All public methods and functions
- ✅ Error handling and edge cases
- ✅ Different parameter combinations
- ✅ Integration with real data files
- ✅ Mocking of external dependencies
- ✅ Both unit and integration test scenarios