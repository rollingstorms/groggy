"""
Groggy - High-performance graph library with memory optimization and Git-like version control

This package provides a Pythonic interface to the high-performance Rust graph library.
"""

# Import directly from Rust extension for Phase 3 functionality
from ._groggy import (
    Graph,
    Subgraph,
    AttrValue,
    NodeFilter,
    EdgeFilter, 
    AttributeFilter,
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
    BaseArray,
    NodesArray,
    EdgesArray,
    MetaNodeArray,
    ComponentsArray,
    SubgraphArray,
    NumArray,
    StatsArray,  # Backward compatibility alias for NumArray
    # Array factory functions for backward compatibility  
    bool_array,
    int_array,
    ones_bool,
    zeros_bool,
    GraphMatrix,
    GraphTable,
    # Table classes
    BaseTable,
    NodesTable,
    EdgesTable,
    # Builder functions with auto-conversion
    array,
    num_array,
    matrix,
    table,
    # Display functionality
    DisplayConfig,
    # Display system
    DisplayConfig,
    TableFormatter,
    # Hierarchical subgraph functionality
    AggregationFunction,
    MetaNode,
    # Visualization system
    VizConfig,
    VizModule,
    InteractiveViz,
    InteractiveVizSession,
    StaticViz,
)

from .types import NodeId, EdgeId, AttrName, StateId, BranchName
from .errors import (
    GroggyError, 
    NodeNotFoundError, 
    EdgeNotFoundError, 
    InvalidInputError, 
    NotImplementedError
)
from ._groggy import parse_node_query, parse_edge_query
# Import neural network submodule
from ._groggy import neural
from . import generators
from . import viz
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
# The classes BaseArray, GraphMatrix, GraphTable are still available for direct instantiation

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

__version__ = "0.4.1"
__all__ = [
    "Graph",
    "Subgraph",
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
    "NodeFilter",
    "EdgeFilter",
    "AttributeFilter", 
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
    "BaseArray",
    "NodesArray",
    "EdgesArray",
    "MetaNodeArray",
    "ComponentsArray",
    "SubgraphArray",
    "NumArray", 
    "StatsArray",  # Backward compatibility
    # Array factory functions  
    "bool_array",
    "int_array",
    "ones_bool", 
    "zeros_bool",
    "GraphMatrix",
    "GraphTable",
    # Table classes
    "BaseTable",
    "NodesTable", 
    "EdgesTable",
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
    "num_array",
    "matrix", 
    "table",
    # Hierarchical subgraph functionality
    "AggregationFunction",
    "MetaNode",
    # Neural network module
    "neural",
    # Visualization system
    "viz",
    "VizConfig",
    "VizModule", 
    "InteractiveViz",
    "InteractiveVizSession",
    "StaticViz",
]

# Apply visualization capabilities to main data structures
# This adds .viz property to all graph and table classes
from .viz import add_viz_accessor

# Add viz accessor to core graph classes
Graph = add_viz_accessor(Graph)
Subgraph = add_viz_accessor(Subgraph)

# Add viz accessor to table classes  
GraphTable = add_viz_accessor(GraphTable)
BaseTable = add_viz_accessor(BaseTable)
NodesTable = add_viz_accessor(NodesTable)
EdgesTable = add_viz_accessor(EdgesTable)

# Add viz accessor to array classes that can be visualized as networks
NodesArray = add_viz_accessor(NodesArray)
EdgesArray = add_viz_accessor(EdgesArray)
SubgraphArray = add_viz_accessor(SubgraphArray)

# Note: Arrays and matrices can be visualized but may need special handling
# for now we focus on the main graph structures
