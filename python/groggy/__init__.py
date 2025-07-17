# groggy/__init__.py
"""
Groggy: A high-performance graph library with Rust backend.

This module provides a clean Python API for graph operations,
backed by a fast Rust implementation with SIMD acceleration.
"""

import json
from typing import Any, List, Union

# Import the Rust backend
try:
    from groggy._core import (
        Graph as _Graph,
        NodeCollection as _NodeCollection,
        EdgeCollection as _EdgeCollection,
    )
except ImportError as e:
    raise ImportError(f"Failed to import Rust backend: {e}")


# Import the complete Graph class from graph.py
from groggy.graph import Graph

# Export main classes
__all__ = ['Graph']
