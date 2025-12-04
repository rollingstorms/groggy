"""
Intermediate Representation (IR) for algorithm optimization.

This module provides data structures and utilities for building and
optimizing algorithm IR before execution.

Phase 1: IR Foundation (Complete)
- Typed IR nodes with domain awareness
- IR graph structure with dependency tracking
- Visualization and analysis utilities
- Dataflow analysis (liveness, dependencies, fusion detection)

Phase 2: IR Optimization (In Progress)
- Dead code elimination (DCE)
- Constant folding
- Common subexpression elimination (CSE)

Future phases will add:
- Loop-invariant code motion (LICM)
- Operation fusion transforms
- JIT compilation
"""

from .analysis import (DataflowAnalysis, DataflowAnalyzer, DependencyChain,
                       LivenessInfo, LoopInfo, analyze_dataflow)
from .graph import IRGraph
from .nodes import (AnyIRNode, AttrIRNode, ControlIRNode, CoreIRNode,
                    GraphIRNode, IRDomain, IRNode)
from .optimizer import IROptimizer, optimize_ir

__all__ = [
    "IRNode",
    "IRDomain",
    "CoreIRNode",
    "GraphIRNode",
    "AttrIRNode",
    "ControlIRNode",
    "AnyIRNode",
    "IRGraph",
    "LivenessInfo",
    "LoopInfo",
    "DependencyChain",
    "DataflowAnalysis",
    "DataflowAnalyzer",
    "analyze_dataflow",
    "IROptimizer",
    "optimize_ir",
]
