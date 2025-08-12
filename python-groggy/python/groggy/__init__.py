"""
Groggy - High-performance graph library with memory optimization and Git-like version control

This package provides a Pythonic interface to the high-performance Rust graph library.
"""

# Import directly from Rust extension for Phase 3 functionality
from ._groggy import (
    Graph,
    AttrValue,
    NodeFilter,
    EdgeFilter, 
    AttributeFilter,
    TraversalResult,
    AggregationResult
)

from .types import NodeId, EdgeId, AttrName, StateId, BranchName
from .errors import (
    GroggyError, 
    NodeNotFoundError, 
    EdgeNotFoundError, 
    InvalidInputError, 
    NotImplementedError
)

__version__ = "0.1.0"
__all__ = [
    "Graph",
    "AttrValue", 
    "NodeId",
    "EdgeId", 
    "AttrName",
    "StateId",
    "BranchName",
    "GroggyError",
    "NodeNotFoundError",
    "EdgeNotFoundError", 
    "InvalidInputError",
    "NotImplementedError",
    # Phase 3 classes
    "NodeFilter",
    "EdgeFilter",
    "AttributeFilter", 
    "TraversalResult",
    "AggregationResult",
]
