"""
Compatibility shim for legacy imports.

The active builder implementation lives under ``groggy.builder`` (package).
This module is retained for backward compatibility and re-exports the
package-based API plus legacy runtime compatibility classes.
"""

from groggy.builder import (AlgorithmBuilder, AttrOps, BuiltAlgorithm,
                            BuiltSampler, CoreOps, GraphHandle, GraphOps,
                            IterOps, LoopContext, SubgraphArrayHandle,
                            SubgraphHandle, VarHandle, algorithm, builder,
                            compiled, traced)
from groggy.builder.legacy_compat import finalize_legacy_loop_steps

__all__ = [
    "AlgorithmBuilder",
    "VarHandle",
    "SubgraphHandle",
    "SubgraphArrayHandle",
    "GraphHandle",
    "LoopContext",
    "CoreOps",
    "GraphOps",
    "AttrOps",
    "IterOps",
    "BuiltAlgorithm",
    "BuiltSampler",
    "builder",
    "algorithm",
    "compiled",
    "traced",
    "finalize_legacy_loop_steps",
]
