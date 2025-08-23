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
    # Statistical arrays and matrices
    GraphArray,
    GraphMatrix,
    GraphTable,
    # Builder functions with auto-conversion
    array,
    matrix,
    table,
    # Display functionality
    DisplayConfig,
    format_array,
    format_matrix,
    format_table,
    format_data_structure,
    detect_display_type,
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
from .generators import (
    complete_graph,
    erdos_renyi,
    barabasi_albert,
    watts_strogatz,
    cycle_graph,
    path_graph,
    star_graph,
    grid_graph,
    tree,
    karate_club,
    social_network,
)
from . import networkx_compat
from .enhanced_query import enhanced_filter_nodes, enhanced_filter_edges
# GraphTable now comes from Rust FFI
from . import table_extensions

# Note: array, matrix, table are now imported directly as auto-converting builder functions
# The classes GraphArray, GraphMatrix, GraphTable are still available for direct instantiation

# Import display formatters for rich display integration
try:
    from .display.formatters import format_array, format_matrix, format_table
    _DISPLAY_AVAILABLE = True
except ImportError:
    _DISPLAY_AVAILABLE = False
    
    # Provide fallback functions if display module is not available
    def format_array(data):
        return f"GraphArray(len={len(data.get('data', []))}, dtype={data.get('dtype', 'object')})"
    
    def format_matrix(data):
        shape = data.get('shape', (0, 0))
        return f"GraphMatrix(shape={shape}, dtype={data.get('dtype', 'object')})"
        
    def format_table(data):
        shape = data.get('shape', (0, 0))
        return f"GraphTable(shape={shape})"

__version__ = "0.3.0"
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
    # Statistical arrays and matrices
    "GraphArray",
    "GraphMatrix",
    "GraphTable",
    # Display functionality  
    "DisplayConfig",
    "format_array",
    "format_matrix", 
    "format_table",
    "format_data_structure",
    "detect_display_type",
    # Graph generators
    "generators",
    "complete_graph",
    "erdos_renyi",
    "barabasi_albert", 
    "watts_strogatz",
    "cycle_graph",
    "path_graph",
    "star_graph",
    "grid_graph",
    "tree",
    "karate_club",
    "social_network",
    # NetworkX interoperability
    "networkx_compat",
    # Enhanced query functions
    "enhanced_filter_nodes",
    "enhanced_filter_edges",
    # Auto-converting builder functions
    "array",
    "matrix", 
    "table",
]
