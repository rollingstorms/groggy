"""
Dataflow analysis passes for IR optimization.

Provides analysis of:
- Data dependencies and control flow
- Variable liveness and lifetimes
- Loop-invariant computations
- Fusion opportunities
- Critical path analysis
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .graph import IRGraph
from .nodes import AnyIRNode, ControlIRNode, IRDomain


@dataclass
class LivenessInfo:
    """
    Track which variables are live at each program point.

    A variable is "live" at a point if its value will be used later
    in the execution. This information is crucial for:
    - Dead code elimination
    - Register allocation / memory reuse
    - Determining when to drop Python references
    """

    # Variables live at entry to this node
    live_in: Set[str] = field(default_factory=set)

    # Variables live at exit from this node
    live_out: Set[str] = field(default_factory=set)

    # Variables defined by this node
    defs: Set[str] = field(default_factory=set)

    # Variables used by this node
    uses: Set[str] = field(default_factory=set)

    # Variables that can be dropped after this node
    can_drop: Set[str] = field(default_factory=set)

    # Variables that might be updated in-place
    can_update_inplace: Set[str] = field(default_factory=set)


@dataclass
class LoopInfo:
    """
    Information about a loop in the IR.

    Tracks:
    - Loop-invariant computations (can be hoisted outside)
    - Loop-carried dependencies (values that flow between iterations)
    - Induction variables
    - Reduction patterns
    """

    # Node ID of the loop control node
    loop_id: str

    # Nodes within the loop body
    body_nodes: List[AnyIRNode] = field(default_factory=list)

    # Variables that don't change within the loop (hoist candidates)
    invariant_vars: Set[str] = field(default_factory=set)

    # Variables that carry values across iterations
    carried_vars: Set[str] = field(default_factory=set)

    # Variables used only for iteration/indexing
    induction_vars: Set[str] = field(default_factory=set)

    # Variables involved in reductions (sum, product, etc.)
    reduction_vars: Set[str] = field(default_factory=set)

    # Nodes that can be hoisted out of the loop
    hoistable_nodes: List[AnyIRNode] = field(default_factory=list)

    # Loop iteration count (if known)
    iteration_count: Optional[int] = None


@dataclass
class DependencyChain:
    """
    A chain of dependent operations that might be fusable.

    Example chains:
    - Arithmetic: a * b + c → fused_madd(a, b, c)
    - Graph ops: transform → neighbor_agg → transform
    - Reductions: map → reduce
    """

    # Nodes in the chain, in execution order
    nodes: List[AnyIRNode] = field(default_factory=list)

    # Primary domain of the chain
    domain: IRDomain = IRDomain.UNKNOWN

    # Pattern type (e.g., "arithmetic_chain", "map_reduce", "neighbor_fusion")
    pattern: str = ""

    # Estimated benefit of fusing (1.0 = fuse, 0.0 = don't)
    fusion_benefit: float = 0.0


@dataclass
class DataflowAnalysis:
    """
    Complete dataflow analysis results for an IR graph.

    Combines liveness analysis, loop analysis, dependency tracking,
    and fusion opportunity detection.
    """

    graph: IRGraph

    # Liveness information per node
    liveness: Dict[str, LivenessInfo] = field(default_factory=dict)

    # Loop information
    loops: List[LoopInfo] = field(default_factory=list)

    # Dependency chains suitable for fusion
    fusion_chains: List[DependencyChain] = field(default_factory=list)

    # Critical path (longest dependency chain)
    critical_path: List[AnyIRNode] = field(default_factory=list)

    # Variables that are never used (dead code)
    dead_vars: Set[str] = field(default_factory=set)

    # Read-after-write dependencies
    raw_deps: Dict[str, List[str]] = field(default_factory=dict)

    # Write-after-read dependencies
    war_deps: Dict[str, List[str]] = field(default_factory=dict)

    # Write-after-write dependencies
    waw_deps: Dict[str, List[str]] = field(default_factory=dict)


class DataflowAnalyzer:
    """
    Performs dataflow analysis on IR graphs.

    Implements various analysis passes:
    - Liveness analysis (backward dataflow)
    - Reaching definitions (forward dataflow)
    - Loop analysis (dominance + natural loops)
    - Dependency classification
    - Fusion opportunity detection
    """

    def __init__(self, graph: IRGraph):
        self.graph = graph
        self.analysis = DataflowAnalysis(graph=graph)

    def analyze(self) -> DataflowAnalysis:
        """
        Run all analysis passes.

        Returns:
            Complete dataflow analysis results
        """
        # Order matters - some analyses depend on previous ones
        self._analyze_dependencies()
        self._analyze_liveness()
        self._analyze_loops()
        self._detect_fusion_chains()
        self._compute_critical_path()

        return self.analysis

    def _analyze_dependencies(self):
        """
        Classify all data dependencies as RAW, WAR, or WAW.

        - RAW (Read-After-Write): true dependency, must preserve order
        - WAR (Write-After-Read): anti-dependency, can sometimes reorder
        - WAW (Write-After-Write): output dependency, can eliminate one
        """
        # Build write and read sets
        writes: Dict[str, List[str]] = defaultdict(list)  # var -> node ids
        reads: Dict[str, List[str]] = defaultdict(list)  # var -> node ids

        for node in self.graph.nodes:
            # Track writes
            if node.output:
                writes[node.output].append(node.id)

            # Track reads
            for input_var in node.inputs:
                reads[input_var].append(node.id)

        # Classify dependencies
        for var in self.graph.var_defs.keys():
            write_nodes = writes.get(var, [])
            read_nodes = reads.get(var, [])

            # RAW: write happens, then read
            if write_nodes and read_nodes:
                self.analysis.raw_deps[var] = read_nodes

            # WAR: read happens, then write (check for redefinition)
            if read_nodes and len(write_nodes) > 1:
                self.analysis.war_deps[var] = write_nodes[1:]

            # WAW: multiple writes to same variable
            if len(write_nodes) > 1:
                self.analysis.waw_deps[var] = write_nodes

    def _analyze_liveness(self):
        """
        Perform backward liveness analysis.

        A variable is live at a point if it will be read later.
        This is computed by iterating backward through the program.
        """
        # Initialize liveness info for each node
        for node in self.graph.nodes:
            info = LivenessInfo()
            info.defs = {node.output} if node.output else set()
            info.uses = set(node.inputs)
            self.analysis.liveness[node.id] = info

        # Iterate backward until fixed point
        nodes = list(reversed(self.graph.topological_order()))
        changed = True
        max_iterations = 100
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for node in nodes:
                info = self.analysis.liveness[node.id]
                old_live_in = info.live_in.copy()

                # live_out = union of live_in of all successors
                info.live_out = set()
                for succ in self.graph.get_dependents(node):
                    succ_info = self.analysis.liveness[succ.id]
                    info.live_out.update(succ_info.live_in)

                # live_in = (live_out - defs) ∪ uses
                info.live_in = (info.live_out - info.defs) | info.uses

                if info.live_in != old_live_in:
                    changed = True

        # Compute additional derived info
        for node in self.graph.nodes:
            info = self.analysis.liveness[node.id]

            # Variables that can be dropped after this node
            # (defined by this node and not live out)
            if node.output:
                if node.output not in info.live_out:
                    info.can_drop.add(node.output)

            # Variables that might be updated in-place
            # (defined here and not live in, or last use)
            if node.output:
                # Check if output is also an input (in-place update pattern)
                if node.output in info.uses:
                    info.can_update_inplace.add(node.output)

        # Find dead variables (never read)
        for var in self.graph.var_defs.keys():
            using_nodes = self.graph.get_using_nodes(var)
            if not using_nodes:
                self.analysis.dead_vars.add(var)

    def _analyze_loops(self):
        """
        Analyze loop structures to find optimization opportunities.

        For each loop:
        - Identify loop-invariant computations (can hoist)
        - Find loop-carried dependencies
        - Detect reduction patterns
        """
        # Find all loop control nodes
        loop_nodes = [
            node
            for node in self.graph.nodes
            if isinstance(node, ControlIRNode)
            and node.op_type in ["loop", "until_converged"]
        ]

        for loop_node in loop_nodes:
            info = LoopInfo(loop_id=loop_node.id)

            # Extract loop metadata
            if loop_node.op_type == "loop":
                info.iteration_count = loop_node.metadata.get("count")

            # Get loop body nodes (stored in metadata)
            body_steps = loop_node.metadata.get("body", [])

            # For now, we track loop info but don't have full body analysis
            # (would need control flow graph to properly analyze nested scopes)
            # This is a placeholder for future enhancement

            # Heuristic: variables defined before loop and used in loop are invariant
            # (assuming no assignments in loop body)

            self.analysis.loops.append(info)

    def _detect_fusion_chains(self):
        """
        Detect chains of operations that could be fused.

        Fusion patterns:
        1. Arithmetic chains: a * b + c → single fused operation
        2. Map-reduce: transform + aggregate
        3. Neighbor operations: transform + neighbor_agg + transform
        """
        visited = set()

        for node in self.graph.topological_order():
            if node.id in visited:
                continue

            # Try to extend a chain from this node
            chain = self._try_build_chain(node, visited)
            if chain and len(chain.nodes) > 1:
                self.analysis.fusion_chains.append(chain)

    def _try_build_chain(
        self, start_node: AnyIRNode, visited: Set[str]
    ) -> Optional[DependencyChain]:
        """
        Try to build a fusion chain starting from a node.

        Args:
            start_node: Node to start chain from
            visited: Set of already-visited node IDs

        Returns:
            DependencyChain if a fusable pattern found, else None
        """
        chain = DependencyChain(nodes=[start_node], domain=start_node.domain)
        visited.add(start_node.id)

        # Extend chain forward while nodes are fusable
        current = start_node
        while True:
            successors = self.graph.get_dependents(current)

            # Chain continues only if there's exactly one successor
            # and it's in the same domain (for now - could relax this)
            if len(successors) != 1:
                break

            next_node = successors[0]

            # Check if fusable
            if not self._can_fuse(current, next_node):
                break

            chain.nodes.append(next_node)
            visited.add(next_node.id)
            current = next_node

        # Classify pattern and estimate benefit
        chain.pattern = self._classify_chain_pattern(chain)
        chain.fusion_benefit = self._estimate_fusion_benefit(chain)

        return chain if len(chain.nodes) > 1 else None

    def _can_fuse(self, node1: AnyIRNode, node2: AnyIRNode) -> bool:
        """Check if two nodes can be fused."""
        # Same domain is easiest to fuse
        if node1.domain != node2.domain:
            return False

        # Core arithmetic ops are highly fusable
        if node1.domain == IRDomain.CORE:
            fusable_ops = {"add", "sub", "mul", "div", "neg", "recip"}
            return node1.op_type in fusable_ops and node2.op_type in fusable_ops

        # Graph ops can sometimes fuse (e.g., pre-transform + neighbor_agg)
        if node1.domain == IRDomain.GRAPH:
            return False  # Conservative for now

        return False

    def _classify_chain_pattern(self, chain: DependencyChain) -> str:
        """Classify what type of fusion pattern this is."""
        if chain.domain == IRDomain.CORE:
            return "arithmetic_chain"
        elif chain.domain == IRDomain.GRAPH:
            return "graph_chain"
        else:
            return "unknown"

    def _estimate_fusion_benefit(self, chain: DependencyChain) -> float:
        """
        Estimate the benefit of fusing this chain.

        Returns:
            Score from 0.0 (don't fuse) to 1.0 (definitely fuse)
        """
        # Simple heuristic: longer chains = more benefit
        # Each fused operation saves an FFI crossing
        num_ops = len(chain.nodes)
        if num_ops <= 1:
            return 0.0

        # Base benefit from eliminating FFI crossings
        base = min(1.0, (num_ops - 1) * 0.25)

        # Bonus for arithmetic (easy to fuse)
        if chain.pattern == "arithmetic_chain":
            base += 0.2

        return min(1.0, base)

    def _compute_critical_path(self):
        """
        Find the critical path (longest dependency chain).

        This identifies the bottleneck that limits parallelism.
        """
        # Compute depth for each node (longest path from root)
        depths: Dict[str, int] = {}
        paths: Dict[str, List[AnyIRNode]] = {}

        for node in self.graph.topological_order():
            # Depth is 1 + max depth of dependencies
            deps = self.graph.get_dependencies(node)
            if not deps:
                depths[node.id] = 0
                paths[node.id] = [node]
            else:
                # Only consider deps that have been processed
                processed_deps = [d for d in deps if d.id in depths]
                if not processed_deps:
                    # No processed deps yet (shouldn't happen in valid topological order)
                    depths[node.id] = 0
                    paths[node.id] = [node]
                else:
                    max_dep_depth = max(depths[dep.id] for dep in processed_deps)
                    depths[node.id] = max_dep_depth + 1

                    # Path is longest path to dep + this node
                    max_dep = max(processed_deps, key=lambda d: depths[d.id])
                    paths[node.id] = paths[max_dep.id] + [node]

        # Critical path is the longest path
        if depths:
            critical_node_id = max(depths.keys(), key=lambda k: depths[k])
            self.analysis.critical_path = paths[critical_node_id]

    def print_analysis(self) -> str:
        """
        Generate human-readable analysis report.

        Returns:
            Multi-line string with analysis results
        """
        lines = [
            "=" * 70,
            "Dataflow Analysis Report",
            "=" * 70,
            f"Graph: {self.graph.name}",
            f"Nodes: {len(self.graph.nodes)}, Variables: {len(self.graph.var_defs)}",
            "",
        ]

        # Dependency summary
        lines.extend(
            [
                "Dependencies:",
                f"  RAW (Read-After-Write): {len(self.analysis.raw_deps)} variables",
                f"  WAR (Write-After-Read): {len(self.analysis.war_deps)} variables",
                f"  WAW (Write-After-Write): {len(self.analysis.waw_deps)} variables",
                "",
            ]
        )

        # Dead code
        if self.analysis.dead_vars:
            lines.extend(
                [
                    f"Dead Variables ({len(self.analysis.dead_vars)}):",
                    "  " + ", ".join(sorted(self.analysis.dead_vars)),
                    "",
                ]
            )

        # Liveness summary
        can_drop_count = sum(
            1 for info in self.analysis.liveness.values() if info.can_drop
        )
        inplace_count = sum(
            1 for info in self.analysis.liveness.values() if info.can_update_inplace
        )

        lines.extend(
            [
                "Liveness Analysis:",
                f"  Nodes with droppable variables: {can_drop_count}",
                f"  Nodes with in-place update opportunities: {inplace_count}",
                "",
            ]
        )

        # Loop analysis
        if self.analysis.loops:
            lines.extend(
                [
                    f"Loops: {len(self.analysis.loops)}",
                ]
            )
            for loop in self.analysis.loops:
                lines.append(f"  Loop {loop.loop_id}:")
                if loop.iteration_count:
                    lines.append(f"    Iterations: {loop.iteration_count}")
                if loop.invariant_vars:
                    lines.append(
                        f"    Invariant vars: {', '.join(loop.invariant_vars)}"
                    )
                if loop.hoistable_nodes:
                    lines.append(f"    Hoistable ops: {len(loop.hoistable_nodes)}")
            lines.append("")

        # Fusion opportunities
        if self.analysis.fusion_chains:
            lines.extend(
                [
                    f"Fusion Opportunities: {len(self.analysis.fusion_chains)}",
                ]
            )
            for i, chain in enumerate(self.analysis.fusion_chains):
                lines.append(
                    f"  {i+1}. {chain.pattern}: {len(chain.nodes)} ops, "
                    f"benefit={chain.fusion_benefit:.2f}"
                )
                for node in chain.nodes:
                    lines.append(f"      - {node}")
            lines.append("")

        # Critical path
        if self.analysis.critical_path:
            lines.extend(
                [
                    f"Critical Path: {len(self.analysis.critical_path)} nodes",
                ]
            )
            for i, node in enumerate(self.analysis.critical_path):
                lines.append(f"  {i+1}. {node}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def analyze_dataflow(graph: IRGraph) -> DataflowAnalysis:
    """
    Convenience function to run dataflow analysis on an IR graph.

    Args:
        graph: IR graph to analyze

    Returns:
        Complete dataflow analysis

    Example:
        >>> analysis = analyze_dataflow(my_graph)
        >>> print(f"Found {len(analysis.fusion_chains)} fusion opportunities")
    """
    analyzer = DataflowAnalyzer(graph)
    return analyzer.analyze()
