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
# viz.py removed - graph visualization now available via graph.graph_viz()
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

# Import comprehensive data import functionality
from .imports import (
    from_csv,
    from_pandas,
    from_numpy,
    from_json,
    from_parquet,
    from_dict,
)

# Provide fallback functions if display module is not available
if not _DISPLAY_AVAILABLE:
    def format_array(data):
        return f"GraphArray(len={len(data.get('data', []))}, dtype={data.get('dtype', 'object')})"

    def format_matrix(data):
        shape = data.get('shape', (0, 0))
        return f"GraphMatrix(shape={shape}, dtype={data.get('dtype', 'object')})"

    def format_table(data):
        shape = data.get('shape', (0, 0))
        return f"GraphTable(shape={shape})"

__version__ = "0.5.1"
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
    # Data import functions
    "from_csv",
    "from_pandas",
    "from_numpy",
    "from_json",
    "from_parquet",
    "from_dict",
    # Hierarchical subgraph functionality
    "AggregationFunction",
    "MetaNode",
    # Neural network module
    "neural",
]

# Apply visualization capabilities to main data structures
# viz.py removed - graph visualization now available via graph.graph_viz() method
# Previous viz accessor system replaced with direct Rust GraphDataSource access
# Graph classes now have built-in graph_viz() method via PySubgraph FFI

# Note: Arrays and matrices can be visualized but may need special handling
# for now we focus on the main graph structures

# Widget auto-loading for production experience (like Plotly)
# DISABLED: Auto-loading can cause JavaScript errors in some environments
# Users can manually enable widgets if needed by calling:
#   from groggy.widgets.widget_loader import auto_load_widget; auto_load_widget()
def _setup_widget_environment():
    """Set up widget environment for seamless Jupyter integration."""
    try:
        # Check if we're in Jupyter
        from IPython import get_ipython
        if get_ipython() is not None:
            # Auto-load widget JavaScript
            from .widgets.widget_loader import auto_load_widget
            auto_load_widget()
    except (ImportError, Exception):
        # Not in Jupyter or widget loading failed - that's fine
        pass

# Auto-setup when groggy is imported (production-ready experience)
# _setup_widget_environment()  # DISABLED - uncomment to enable auto-loading

# Viz accessors are now implemented directly in Rust FFI

def _jupyter_labextension_paths():
    """
    Discovery hook for JupyterLab federated extensions.
    
    This tells JupyterLab where to find our widget extension.
    """
    return [{
        "src": "labextension",  # Directory inside the groggy package
        "dest": "groggy-widgets",  # Must match NPM package name and Python _model_module
    }]
