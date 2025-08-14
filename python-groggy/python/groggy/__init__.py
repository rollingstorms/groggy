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
    AggregationResult,
    GroupedAggregationResult,
    PyResultHandle,
    PyAttributeCollection,
    # Version control classes
    Commit,
    BranchInfo,
    HistoryStatistics,
    HistoricalView,
)

from .types import NodeId, EdgeId, AttrName, StateId, BranchName
from .errors import (
    GroggyError, 
    NodeNotFoundError, 
    EdgeNotFoundError, 
    InvalidInputError, 
    NotImplementedError
)
from .query_parser import parse_node_query, parse_edge_query
from . import generators
from . import networkx_compat
from .enhanced_query import enhanced_filter_nodes, enhanced_filter_edges
from .graph_table import GraphTable
from . import table_extensions

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
    "GroupedAggregationResult",
    "PyResultHandle",
    "PyAttributeCollection",
    # Version control classes
    "Commit",
    "BranchInfo", 
    "HistoryStatistics",
    "HistoricalView",
    # Graph generators
    "generators",
    # NetworkX interoperability
    "networkx_compat",
    # Enhanced query functions
    "enhanced_filter_nodes",
    "enhanced_filter_edges",
    # Graph table functionality
    "GraphTable",
]
