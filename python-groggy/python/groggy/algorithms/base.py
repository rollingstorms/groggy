"""
Algorithm handle system for Groggy.

Provides base classes and factory functions for working with algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from groggy import _groggy


class AlgorithmHandle(ABC):
    """Base class for algorithm handles."""

    @abstractmethod
    def to_spec(self) -> Dict[str, Any]:
        """Convert the algorithm handle to a pipeline spec entry."""
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Get the algorithm identifier."""
        pass


class RustAlgorithmHandle(AlgorithmHandle):
    """Handle for pre-registered Rust algorithms."""

    def __init__(self, algorithm_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Create a handle for a Rust algorithm.

        Args:
            algorithm_id: The registered algorithm ID (e.g., "centrality.pagerank")
            params: Optional parameters for the algorithm
        """
        self._id = algorithm_id
        self._params = params or {}

        # Validate algorithm exists
        try:
            metadata = _groggy.pipeline.get_algorithm_metadata(algorithm_id)
            self._metadata = metadata
        except RuntimeError as e:
            raise ValueError(f"Algorithm '{algorithm_id}' not found: {e}")

    @property
    def id(self) -> str:
        """Get the algorithm identifier."""
        return self._id

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata."""
        return {k: v.value for k, v in self._metadata.items()}

    def with_params(self, **params) -> "RustAlgorithmHandle":
        """
        Create a new handle with updated parameters.

        Args:
            **params: Parameters to update

        Returns:
            New handle with updated parameters
        """
        new_params = {**self._params, **params}
        return RustAlgorithmHandle(self._id, new_params)

    def validate(self) -> bool:
        """
        Validate the current parameters.

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Convert params to AttrValues for validation
        attr_params = {}
        for key, value in self._params.items():
            if not isinstance(value, _groggy.AttrValue):
                # Auto-wrap common types
                attr_params[key] = _groggy.AttrValue(value)
            else:
                attr_params[key] = value

        errors = _groggy.pipeline.validate_algorithm_params(self._id, attr_params)
        if errors:
            raise ValueError(f"Parameter validation failed:\n  " + "\n  ".join(errors))
        return True

    def to_spec(self) -> Dict[str, Any]:
        """Convert to pipeline spec entry."""
        # Ensure parameters are wrapped in AttrValue
        wrapped_params = {}
        for key, value in self._params.items():
            if not isinstance(value, _groggy.AttrValue):
                wrapped_params[key] = _groggy.AttrValue(value)
            else:
                wrapped_params[key] = value

        return {"id": self._id, "params": wrapped_params}

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"RustAlgorithmHandle('{self._id}', {params_str})"


def algorithm(algorithm_id: str, **params) -> RustAlgorithmHandle:
    """
    Create an algorithm handle.

    Args:
        algorithm_id: The registered algorithm ID
        **params: Algorithm parameters

    Returns:
        Algorithm handle ready for use in pipelines

    Example:
        >>> import groggy.algorithms as alg
        >>> pagerank = alg.algorithm("centrality.pagerank", max_iter=20, damping=0.85)
        >>> result = subgraph.apply(pagerank)
    """
    return RustAlgorithmHandle(algorithm_id, params)
