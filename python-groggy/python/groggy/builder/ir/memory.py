"""
Memory optimization analysis for IR graphs.

Provides memory reuse analysis, in-place operation detection, and allocation tracking
to minimize memory footprint during algorithm execution.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .analysis import DataflowAnalysis
from .graph import IRGraph
from .nodes import IRDomain, IRNode


@dataclass
class MemoryAllocation:
    """Represents a memory allocation for a variable."""

    var_name: str
    size_estimate: int  # Number of elements
    element_type: str  # "float", "int", "bool"

    def bytes(self) -> int:
        """Estimate memory usage in bytes."""
        type_sizes = {"float": 8, "int": 8, "bool": 1}
        return self.size_estimate * type_sizes.get(self.element_type, 8)


@dataclass
class InPlaceCandidate:
    """Represents an operation that can potentially be done in-place."""

    node: IRNode
    input_var: str  # Variable that can be overwritten
    output_var: str  # Output variable
    reason: str  # Why this is in-place capable


@dataclass
class BufferReuseOpportunity:
    """Represents an opportunity to reuse memory between variables."""

    dead_var: str  # Variable whose memory is no longer needed
    new_var: str  # Variable that can reuse the memory
    node: IRNode  # Node that creates new_var
    size_compatible: bool  # Whether sizes are compatible


class MemoryAnalysis:
    """
    Analyzes memory usage and identifies optimization opportunities.

    Features:
    - Tracks variable lifetimes and memory allocations
    - Identifies in-place operation opportunities
    - Finds buffer reuse opportunities
    - Estimates peak memory usage
    """

    def __init__(self, ir_graph: IRGraph, node_count: Optional[int] = None):
        """
        Initialize memory analysis.

        Args:
            ir_graph: The IR graph to analyze
            node_count: Number of nodes in the graph (for size estimation)
        """
        self.ir_graph = ir_graph
        self.node_count = node_count or 10000  # Default size estimate

        # Run dataflow analysis
        from .analysis import DataflowAnalyzer

        analyzer = DataflowAnalyzer(ir_graph)
        self.dataflow = analyzer.analyze()

        # Analysis results
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.in_place_candidates: List[InPlaceCandidate] = []
        self.reuse_opportunities: List[BufferReuseOpportunity] = []
        self.peak_memory_bytes: int = 0

        # Run analysis
        self._analyze()

    def _analyze(self):
        """Run all memory analysis passes."""
        self._track_allocations()
        self._find_in_place_candidates()
        self._find_reuse_opportunities()
        self._estimate_peak_memory()

    def _track_allocations(self):
        """Track all memory allocations in the IR graph."""
        for node in self.ir_graph.nodes:
            if node.output:  # Some nodes may not have outputs (e.g., control flow)
                # Skip constants - they don't allocate arrays
                if node.op_type == "constant":
                    continue

                # Estimate size and type based on operation
                size = self._estimate_variable_size(node, node.output)
                elem_type = self._infer_element_type(node, node.output)

                self.allocations[node.output] = MemoryAllocation(
                    var_name=node.output, size_estimate=size, element_type=elem_type
                )

    def _estimate_variable_size(self, node: IRNode, var_name: str) -> int:
        """Estimate the size of a variable based on the operation that creates it."""
        # Graph-level operations typically produce node-sized arrays
        if node.domain == IRDomain.GRAPH:
            if node.op_type in ["degree", "neighbor_agg"]:
                return self.node_count
            elif node.op_type == "subgraph":
                return self.node_count // 2  # Rough estimate

        # Core operations preserve size from inputs
        elif node.domain == IRDomain.CORE:
            if node.inputs:
                first_input = node.inputs[0]
                if first_input in self.allocations:
                    return self.allocations[first_input].size_estimate
            return self.node_count  # Default

        # Control operations
        elif node.domain == IRDomain.CONTROL:
            return 0  # No allocation

        return self.node_count  # Default

    def _infer_element_type(self, node: IRNode, var_name: str) -> str:
        """Infer the element type of a variable."""
        # Comparison operations produce booleans
        if node.op_type in ["compare", "eq", "lt", "gt", "le", "ge"]:
            return "bool"

        # Degree operations produce integers
        if node.op_type == "degree":
            return "int"

        # Most operations produce floats
        return "float"

    def _find_in_place_candidates(self):
        """Find operations that can be performed in-place."""
        liveness_info = self.dataflow.liveness

        for node in self.ir_graph.nodes:
            # Check if this is an in-place capable operation
            if not self._is_in_place_capable(node):
                continue

            # For binary operations, check if we can overwrite the first input
            if node.inputs and node.output:
                input_var = node.inputs[0]
                output_var = node.output

                # Check if input is no longer live after this operation
                # (except through the output)
                if self._can_overwrite_v2(
                    node.id, input_var, output_var, liveness_info
                ):
                    self.in_place_candidates.append(
                        InPlaceCandidate(
                            node=node,
                            input_var=input_var,
                            output_var=output_var,
                            reason=self._get_in_place_reason(node),
                        )
                    )

    def _is_in_place_capable(self, node: IRNode) -> bool:
        """Check if an operation can be performed in-place."""
        # Arithmetic operations can be in-place
        if node.op_type in ["add", "sub", "mul", "div", "pow"]:
            return True

        # Some element-wise operations
        if node.op_type in ["recip", "sqrt", "abs", "neg"]:
            return True

        # Conditional where with same input/output size
        if node.op_type == "where":
            return True

        return False

    def _can_overwrite_v2(
        self, node_id: str, input_var: str, output_var: str, liveness_info: Dict
    ) -> bool:
        """Check if input variable can be safely overwritten."""
        # Check if input dies at this node (not used afterwards)
        # This is conservative: we check if input appears in live_out of this node
        if node_id in liveness_info:
            live_out = liveness_info[node_id].live_out
            # If input is still live after this operation, we can't overwrite it
            # unless it's only live through the output (aliasing)
            if input_var in live_out and input_var != output_var:
                return False

        return True

    def _get_in_place_reason(self, node: IRNode) -> str:
        """Get human-readable reason for in-place capability."""
        if node.op_type in ["add", "sub", "mul", "div"]:
            return f"Element-wise {node.op_type} can modify first operand in-place"
        elif node.op_type in ["recip", "sqrt", "abs", "neg"]:
            return f"Unary {node.op_type} can operate in-place"
        elif node.op_type == "where":
            return "Conditional assignment can be in-place when input dies"
        return "In-place capable"

    def _find_reuse_opportunities(self):
        """Find opportunities to reuse memory buffers."""
        liveness_info = self.dataflow.liveness
        execution_order = [node.id for node in self.ir_graph.nodes]  # Order from list

        # Track which variables are dead at each point
        for i, node_id in enumerate(execution_order):
            node = self.ir_graph.node_map[node_id]

            # Check live_out to see which variables are dead after this node
            if node_id in liveness_info:
                live_out = liveness_info[node_id].live_out

                # Find variables that were live before but are dead now
                previous_live = set()
                if i > 0:
                    prev_id = execution_order[i - 1]
                    if prev_id in liveness_info:
                        previous_live = liveness_info[prev_id].live_out

                newly_dead = previous_live - live_out

                # For the output of this node, check if we can reuse a dead buffer
                if node.output:
                    for dead_var in newly_dead:
                        if self._buffers_compatible(dead_var, node.output):
                            self.reuse_opportunities.append(
                                BufferReuseOpportunity(
                                    dead_var=dead_var,
                                    new_var=node.output,
                                    node=node,
                                    size_compatible=True,
                                )
                            )

    def _buffers_compatible(self, var1: str, var2: str) -> bool:
        """Check if two variables can share the same buffer."""
        if var1 not in self.allocations or var2 not in self.allocations:
            return False

        alloc1 = self.allocations[var1]
        alloc2 = self.allocations[var2]

        # Size must be compatible (exact match or var2 is smaller)
        if alloc2.size_estimate > alloc1.size_estimate:
            return False

        # Type must match (can't reuse int buffer for float)
        if alloc1.element_type != alloc2.element_type:
            return False

        return True

    def _estimate_peak_memory(self):
        """Estimate peak memory usage during execution."""
        liveness_info = self.dataflow.liveness

        max_live_bytes = 0

        for node_id, liveness in liveness_info.items():
            # Sum up memory for all live variables at this point
            live_bytes = sum(
                self.allocations[var].bytes()
                for var in liveness.live_out
                if var in self.allocations
            )
            max_live_bytes = max(max_live_bytes, live_bytes)

        self.peak_memory_bytes = max_live_bytes

    def get_summary(self) -> Dict:
        """Get a summary of memory analysis results."""
        total_allocations = sum(a.bytes() for a in self.allocations.values())

        return {
            "total_variables": len(self.allocations),
            "total_allocated_bytes": total_allocations,
            "total_allocated_mb": total_allocations / (1024 * 1024),
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_mb": self.peak_memory_bytes / (1024 * 1024),
            "memory_efficiency": (
                (self.peak_memory_bytes / total_allocations * 100)
                if total_allocations > 0
                else 0
            ),
            "in_place_candidates": len(self.in_place_candidates),
            "reuse_opportunities": len(self.reuse_opportunities),
            "potential_savings_bytes": self._estimate_savings(),
            "potential_savings_mb": self._estimate_savings() / (1024 * 1024),
        }

    def _estimate_savings(self) -> int:
        """Estimate potential memory savings from optimizations."""
        savings = 0

        # Savings from in-place operations
        for candidate in self.in_place_candidates:
            if candidate.output_var in self.allocations:
                savings += self.allocations[candidate.output_var].bytes()

        # Savings from buffer reuse
        for opp in self.reuse_opportunities:
            if opp.new_var in self.allocations:
                savings += self.allocations[opp.new_var].bytes()

        # Avoid double-counting
        return savings // 2

    def print_report(self):
        """Print a human-readable memory analysis report."""
        summary = self.get_summary()

        print("=" * 80)
        print("MEMORY ANALYSIS REPORT")
        print("=" * 80)

        print(f"\n📊 Memory Statistics:")
        print(f"  Total variables: {summary['total_variables']}")
        print(f"  Total allocated: {summary['total_allocated_mb']:.2f} MB")
        print(f"  Peak usage: {summary['peak_memory_mb']:.2f} MB")
        print(f"  Memory efficiency: {summary['memory_efficiency']:.1f}%")

        print(f"\n🔧 Optimization Opportunities:")
        print(f"  In-place candidates: {summary['in_place_candidates']}")
        print(f"  Buffer reuse opportunities: {summary['reuse_opportunities']}")
        print(f"  Potential savings: {summary['potential_savings_mb']:.2f} MB")

        if self.in_place_candidates:
            print(f"\n✅ In-Place Operation Candidates:")
            for i, candidate in enumerate(self.in_place_candidates[:5], 1):
                alloc = self.allocations.get(candidate.output_var)
                size_str = f"{alloc.bytes() / 1024:.1f} KB" if alloc else "unknown"
                print(
                    f"  {i}. {candidate.node.op_type}: {candidate.input_var} → {candidate.output_var}"
                )
                print(f"     Save: {size_str}, Reason: {candidate.reason}")
            if len(self.in_place_candidates) > 5:
                print(f"  ... and {len(self.in_place_candidates) - 5} more")

        if self.reuse_opportunities:
            print(f"\n♻️  Buffer Reuse Opportunities:")
            for i, opp in enumerate(self.reuse_opportunities[:5], 1):
                alloc = self.allocations.get(opp.new_var)
                size_str = f"{alloc.bytes() / 1024:.1f} KB" if alloc else "unknown"
                print(f"  {i}. Reuse {opp.dead_var} for {opp.new_var}")
                print(f"     Save: {size_str}, Op: {opp.node.op_type}")
            if len(self.reuse_opportunities) > 5:
                print(f"  ... and {len(self.reuse_opportunities) - 5} more")

        print("\n" + "=" * 80)


def analyze_memory(
    ir_graph: IRGraph, node_count: Optional[int] = None
) -> MemoryAnalysis:
    """
    Convenience function to analyze memory usage of an IR graph.

    Args:
        ir_graph: The IR graph to analyze
        node_count: Number of nodes in the graph (for size estimation)

    Returns:
        MemoryAnalysis object with results
    """
    return MemoryAnalysis(ir_graph, node_count)
