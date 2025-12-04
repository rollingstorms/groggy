"""
Algorithm Builder DSL for composing custom algorithms from steps.

This module provides a high-level interface for building custom algorithms
by composing pre-registered steps, with operator overloading for natural
mathematical syntax.

Example (new syntax with operators):
    >>> from groggy.builder import algorithm
    >>>
    >>> @algorithm("pagerank")
    ... def pagerank(sG, damping=0.85, max_iter=100):
    ...     ranks = sG.nodes(1.0 / sG.N)
    ...     deg = ranks.degrees()
    ...
    ...     with sG.iterate(max_iter):
    ...         neighbor_sum = sG @ (ranks / (deg + 1e-9))
    ...         ranks = sG.var("ranks", damping * neighbor_sum + (1 - damping) / sG.N)
    ...
    ...     return ranks.normalize()
    >>>
    >>> pr = pagerank(max_iter=50)
    >>> result = subgraph.apply(pr)

Example (original syntax, backward compatible):
    >>> from groggy.builder import AlgorithmBuilder
    >>>
    >>> builder = AlgorithmBuilder("my_algorithm")
    >>> nodes = builder.init_nodes(default=0.0)
    >>> degrees = builder.node_degrees(nodes)
    >>> normalized = builder.normalize(degrees)
    >>> builder.attach_as("degree_normalized", normalized)
    >>>
    >>> algo = builder.build()
    >>> result = subgraph.apply(algo)
"""

from groggy.builder.algorithm_builder import (AlgorithmBuilder, BuiltAlgorithm,
                                              LoopContext, builder)
from groggy.builder.decorators import algorithm, compiled, traced
from groggy.builder.traits.attr import AttrOps
from groggy.builder.traits.core import CoreOps
from groggy.builder.traits.graph import GraphOps
from groggy.builder.traits.iter import IterOps
from groggy.builder.varhandle import GraphHandle, SubgraphHandle, VarHandle

# Export main classes
__all__ = [
    "AlgorithmBuilder",
    "VarHandle",
    "SubgraphHandle",
    "GraphHandle",
    "LoopContext",
    "CoreOps",
    "GraphOps",
    "AttrOps",
    "IterOps",
    "BuiltAlgorithm",
    "builder",
    "algorithm",
    "compiled",
    "traced",
]
