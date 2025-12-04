"""
Tests for the examples module.

These tests verify that the documented examples work correctly.
"""

import pytest

from groggy.examples import (example_algorithm_discovery,
                             example_algorithm_reuse, example_error_handling,
                             example_multi_algorithm_pipeline,
                             example_parameter_customization,
                             example_single_algorithm)


def test_example_single_algorithm():
    """Test the single algorithm example."""
    result = example_single_algorithm()
    assert result is not None
    assert len(result.nodes) == 20


def test_example_multi_algorithm_pipeline():
    """Test the multi-algorithm pipeline example."""
    result = example_multi_algorithm_pipeline()
    assert result is not None
    assert len(result.nodes) == 50


def test_example_algorithm_discovery():
    """Test the algorithm discovery example."""
    info = example_algorithm_discovery()
    assert isinstance(info, dict)
    assert "description" in info
    assert "version" in info


def test_example_parameter_customization():
    """Test the parameter customization example."""
    results = example_parameter_customization()
    assert len(results) == 3
    assert all(r is not None for r in results)


def test_example_error_handling():
    """Test the error handling example."""
    # Should not raise - errors are caught internally
    example_error_handling()


def test_example_algorithm_reuse():
    """Test the algorithm reuse example."""
    results = example_algorithm_reuse()
    assert len(results) == 3
    assert all(r is not None for r in results)
