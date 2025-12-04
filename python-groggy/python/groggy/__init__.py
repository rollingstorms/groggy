"""
Groggy - High-performance graph library with memory optimization and Git-like version control

This package provides a Pythonic interface to the high-performance Rust graph library.
"""

# GraphTable now comes from Rust FFI
# Import directly from Rust extension for Phase 3 functionality
from ._groggy import StatsArray  # Backward compatibility alias for NumArray
from ._groggy import (  # Version control classes; Statistical arrays and matrices; Array factory functions for backward compatibility; Table classes; Builder functions with auto-conversion; Display functionality; Display system; Hierarchical subgraph functionality
    AggregationFunction, AggregationResult, AttributeFilter, AttrValue,
    BaseArray, BaseTable, BranchInfo, Commit, ComponentsArray, DisplayConfig,
    EdgeFilter, EdgesArray, EdgesTable, Graph, GraphMatrix, GraphTable,
    GroupedAggregationResult, HistoricalView, HistoryStatistics, MetaNode,
    MetaNodeArray, NodeFilter, NodesArray, NodesTable, NumArray,
    PyAttributeCollection, PyResultHandle, Subgraph, SubgraphArray,
    TableFormatter, array, bool_array, int_array, matrix, neural, num_array,
    ones_bool, parse_edge_query, parse_node_query, table, zeros_bool)
from .enhanced_query import enhanced_filter_edges, enhanced_filter_nodes
from .errors import (EdgeNotFoundError, GroggyError, InvalidInputError,
                     NodeNotFoundError, NotImplementedError)
# Import module-level after Graph is defined to avoid circular import
from . import generators, networkx_compat, table_extensions
# viz.py removed - graph visualization now available via graph.graph_viz()
from .generators import (barabasi_albert, complete_graph, cycle_graph,
                         erdos_renyi, grid_graph, karate_club, path_graph,
                         social_network, star_graph, tree, watts_strogatz)
from .types import AttrName, BranchName, EdgeId, NodeId, StateId

# Note: array, matrix, table are now imported directly as auto-converting builder functions
# The classes BaseArray, GraphMatrix, GraphTable are still available for direct instantiation

# Import display formatters for rich display integration
try:
    from .display.formatters import format_array, format_matrix, format_table

    _DISPLAY_AVAILABLE = True
except ImportError:
    _DISPLAY_AVAILABLE = False

from . import algorithms
# Import builder DSL (Phase 5)
from .builder import AlgorithmBuilder, VarHandle, builder
# Import comprehensive data import functionality
from .imports import (from_csv, from_dict, from_json, from_numpy, from_pandas,
                      from_parquet)
# Import algorithm and pipeline APIs (Phase 4)
from .pipeline import Pipeline, apply, pipeline, print_profile

# Provide fallback functions if display module is not available
if not _DISPLAY_AVAILABLE:

    def format_array(data):
        return f"GraphArray(len={len(data.get('data', []))}, dtype={data.get('dtype', 'object')})"

    def format_matrix(data):
        shape = data.get("shape", (0, 0))
        return f"GraphMatrix(shape={shape}, dtype={data.get('dtype', 'object')})"

    def format_table(data):
        shape = data.get("shape", (0, 0))
        return f"GraphTable(shape={shape})"


__version__ = "0.5.1"
TemporalSnapshot = _groggy.TemporalSnapshot
ExistenceIndex = _groggy.ExistenceIndex

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
    # Algorithm and pipeline APIs (Phase 4)
    "pipeline",
    "Pipeline",
    "apply",
    "print_profile",
    "algorithms",
    # Builder DSL (Phase 5)
    "builder",
    "AlgorithmBuilder",
    "VarHandle",
    "TemporalSnapshot",
    "ExistenceIndex",
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


# Add convenience method to Subgraph for method chaining
def _subgraph_apply(self, algorithm_or_pipeline, persist=True, return_profile=False):
    """
    Apply an algorithm or pipeline to this subgraph.

    Convenience method for fluent method chaining.

    Args:
        algorithm_or_pipeline: AlgorithmHandle, Pipeline, or list of algorithms
        persist: Whether to persist algorithm results as attributes (default: True)
        return_profile: If True, return (subgraph, profile_dict); otherwise just subgraph (default: False)

    Returns:
        Processed subgraph with algorithm results (or tuple with profile if return_profile=True)

    Example:
        >>> result = sg.apply(algorithms.centrality.pagerank()).table()
        >>> result = sg.apply([algo1, algo2, algo3]).viz.draw()
        >>> result, profile = sg.apply(algo, return_profile=True)
    """
    return apply(
        self, algorithm_or_pipeline, persist=persist, return_profile=return_profile
    )


# Monkey-patch the method onto Subgraph
Subgraph.apply = _subgraph_apply


# Monkey-patch a direct apply method onto Graph to avoid expensive __getattr__ delegation
def _graph_apply(self, algorithm_or_pipeline, persist=True, return_profile=False):
    """
    Apply an algorithm or pipeline to this graph.

    Creates a full-graph subgraph view and delegates to Subgraph.apply.
    This direct method avoids the overhead of Python's __getattr__ delegation.

    Args:
        algorithm_or_pipeline: AlgorithmHandle, Pipeline, or list of algorithms
        persist: Whether to persist algorithm results as attributes (default: True)
        return_profile: If True, return (subgraph, profile_dict); otherwise just subgraph (default: False)

    Returns:
        Processed subgraph with algorithm results (or tuple with profile if return_profile=True)

    Example:
        >>> result = g.apply(algorithms.centrality.pagerank())
        >>> result, profile = g.apply(algo, return_profile=True)
    """
    # Use the efficient Rust method to create a full subgraph
    full_subgraph = self.to_subgraph()
    return apply(
        full_subgraph,
        algorithm_or_pipeline,
        persist=persist,
        return_profile=return_profile,
    )


Graph.apply = _graph_apply


def _jupyter_labextension_paths():
    """
    Discovery hook for JupyterLab federated extensions.

    This tells JupyterLab where to find our widget extension.
    """
    return [
        {
            "src": "labextension",  # Directory inside the groggy package
            "dest": "groggy-widgets",  # Must match NPM package name and Python _model_module
        }
    ]
