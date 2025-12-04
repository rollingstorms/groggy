"""
Parallel execution analysis and planning for IR graphs.

Detects independent operations that can be executed in parallel,
identifies data-parallel operations, and generates parallel execution plans.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .batch import BatchExecutionPlan, BatchPlanGenerator
from .graph import IRGraph
from .nodes import AnyIRNode, IRDomain


@dataclass
class ParallelGroup:
    """
    A group of operations that can execute in parallel.

    Operations in the same group:
    - Have no data dependencies on each other
    - Can be executed concurrently
    - May share inputs but must have distinct outputs
    """

    # Node IDs in this parallel group
    node_ids: List[str] = field(default_factory=list)

    # Shared input variables (must be read-only)
    shared_inputs: Set[str] = field(default_factory=set)

    # Output variables (must be distinct)
    outputs: Set[str] = field(default_factory=set)

    # Group dependencies (groups that must execute before this one)
    depends_on: Set[int] = field(default_factory=set)

    # Estimated parallelism benefit (1.0 = no benefit, >1.0 = speedup)
    parallelism_factor: float = 1.0


@dataclass
class ParallelExecutionPlan:
    """
    An execution plan that exploits parallelism.

    Organizes operations into parallel groups that can be executed
    concurrently, maximizing throughput on multi-core systems.
    """

    # Parallel groups in execution order
    groups: List[ParallelGroup] = field(default_factory=list)

    # Fallback sequential plan
    sequential_plan: Optional[BatchExecutionPlan] = None

    # Estimated speedup from parallelization
    estimated_speedup: float = 1.0

    # Whether to use parallel execution
    use_parallel: bool = True

    def to_json(self) -> str:
        """Serialize to JSON for FFI"""
        import json

        return json.dumps(
            {
                "groups": [
                    {
                        "node_ids": g.node_ids,
                        "shared_inputs": list(g.shared_inputs),
                        "outputs": list(g.outputs),
                        "depends_on": list(g.depends_on),
                        "parallelism_factor": g.parallelism_factor,
                    }
                    for g in self.groups
                ],
                "sequential_plan": (
                    self.sequential_plan.to_json() if self.sequential_plan else None
                ),
                "estimated_speedup": self.estimated_speedup,
                "use_parallel": self.use_parallel,
            }
        )


class ParallelAnalyzer:
    """
    Analyzes IR graphs to detect parallelization opportunities.

    The analyzer:
    1. Builds dependency graph
    2. Identifies independent operations at each level
    3. Groups operations by execution stage
    4. Detects data-parallel operations
    5. Estimates parallelization benefit
    """

    def __init__(self, ir_graph: IRGraph):
        self.ir_graph = ir_graph
        self.dependency_levels: List[List[str]] = []
        self.node_dependencies: Dict[str, Set[str]] = {}
        self.node_dependents: Dict[str, Set[str]] = {}

    def analyze(self) -> ParallelExecutionPlan:
        """
        Analyze the IR graph and generate a parallel execution plan.

        Returns:
            ParallelExecutionPlan with detected parallelism opportunities
        """
        # Step 1: Build dependency graph
        self._build_dependency_graph()

        # Step 2: Compute execution levels (operations at same level can run in parallel)
        self._compute_execution_levels()

        # Step 3: Group operations into parallel groups
        groups = self._create_parallel_groups()

        # Step 4: Estimate speedup
        estimated_speedup = self._estimate_speedup(groups)

        # Step 5: Generate fallback sequential plan
        batch_gen = BatchPlanGenerator(self.ir_graph)
        sequential_plan = batch_gen.generate()

        return ParallelExecutionPlan(
            groups=groups,
            sequential_plan=sequential_plan,
            estimated_speedup=estimated_speedup,
            use_parallel=estimated_speedup > 1.2,  # Only use parallel if >20% benefit
        )

    def _build_dependency_graph(self):
        """
        Build the dependency graph: which nodes depend on which other nodes.

        A node B depends on node A if:
        - B uses a variable that A produces
        - B has side effects that must execute after A
        """
        self.node_dependencies = defaultdict(set)
        self.node_dependents = defaultdict(set)

        # Map variable to producing node
        var_to_producer: Dict[str, str] = {}

        for node in self.ir_graph.nodes:
            # Record this node as producer of its output
            if node.output:
                var_to_producer[node.output] = node.id

            # Find dependencies based on inputs
            for input_var in node.inputs:
                if input_var in var_to_producer:
                    producer_id = var_to_producer[input_var]
                    self.node_dependencies[node.id].add(producer_id)
                    self.node_dependents[producer_id].add(node.id)

    def _compute_execution_levels(self):
        """
        Compute execution levels using topological sort with level assignment.

        Operations at the same level have no dependencies on each other
        and can be executed in parallel.
        """
        # Count dependencies for each node
        in_degree = {
            node.id: len(self.node_dependencies[node.id])
            for node in self.ir_graph.nodes
        }

        # Start with nodes that have no dependencies
        current_level = [
            node.id for node in self.ir_graph.nodes if in_degree[node.id] == 0
        ]

        while current_level:
            self.dependency_levels.append(current_level)
            next_level = []

            for node_id in current_level:
                # Reduce in-degree for all dependents
                for dependent_id in self.node_dependents[node_id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_level.append(dependent_id)

            current_level = next_level

    def _create_parallel_groups(self) -> List[ParallelGroup]:
        """
        Create parallel groups from execution levels.

        Each level becomes a parallel group, potentially subdivided if
        operations have different characteristics (e.g., graph ops vs arithmetic).
        """
        groups = []

        for level_idx, level_nodes in enumerate(self.dependency_levels):
            # All nodes at this level can run in parallel
            node_objs = [self.ir_graph.node_map[nid] for nid in level_nodes]

            # Collect shared inputs and outputs
            shared_inputs = set()
            outputs = set()
            for node in node_objs:
                shared_inputs.update(node.inputs)
                if node.output:
                    outputs.add(node.output)

            # Determine dependencies (all previous groups)
            depends_on = set(range(level_idx))

            # Estimate parallelism factor (how much speedup from parallelizing this group)
            parallelism_factor = self._estimate_group_parallelism(node_objs)

            group = ParallelGroup(
                node_ids=level_nodes,
                shared_inputs=shared_inputs,
                outputs=outputs,
                depends_on=depends_on,
                parallelism_factor=parallelism_factor,
            )
            groups.append(group)

        return groups

    def _estimate_group_parallelism(self, nodes: List[AnyIRNode]) -> float:
        """
        Estimate the parallelism factor for a group of nodes.

        Factors considered:
        - Number of nodes (more nodes = more parallelism)
        - Operation types (data-parallel ops benefit more)
        - Operation cost (heavy ops benefit more from parallelization)

        Returns:
            Estimated speedup factor (1.0 = sequential, >1.0 = parallel benefit)
        """
        if len(nodes) <= 1:
            return 1.0

        # Base parallelism from number of operations
        num_ops = len(nodes)
        base_factor = min(num_ops, 8)  # Cap at 8 (typical core count)

        # Check if operations are data-parallel (element-wise)
        data_parallel_ops = {
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "abs",
            "compare",
            "where",
            "map",
            "element_wise",
        }

        parallel_count = sum(1 for node in nodes if node.op_type in data_parallel_ops)
        parallel_ratio = parallel_count / num_ops if num_ops > 0 else 0.0

        # Data-parallel operations benefit more from parallelization
        parallelism_factor = 1.0 + (base_factor - 1.0) * (0.5 + 0.5 * parallel_ratio)

        # Adjust for operation cost
        # Heavy operations (neighbor aggregation, reductions) benefit more
        heavy_ops = {"neighbor_agg", "reduce", "degree", "connected_components"}
        heavy_count = sum(1 for node in nodes if node.op_type in heavy_ops)

        if heavy_count > 0:
            # Boost parallelism factor for heavy operations
            parallelism_factor *= 1.5

        return parallelism_factor

    def _estimate_speedup(self, groups: List[ParallelGroup]) -> float:
        """
        Estimate overall speedup from parallel execution.

        Uses Amdahl's law: speedup depends on fraction of parallelizable work
        and parallelism factor for that work.
        """
        if not groups:
            return 1.0

        # Calculate weighted speedup
        total_ops = sum(len(g.node_ids) for g in groups)
        if total_ops == 0:
            return 1.0

        weighted_speedup = 0.0
        for group in groups:
            weight = len(group.node_ids) / total_ops
            weighted_speedup += weight * group.parallelism_factor

        return weighted_speedup

    def print_parallelism_report(self):
        """Print a human-readable parallelism analysis report."""
        if not self.dependency_levels:
            self._build_dependency_graph()
            self._compute_execution_levels()

        print("=== Parallel Execution Analysis ===\n")
        print(f"Total nodes: {len(self.ir_graph.nodes)}")
        print(f"Execution levels: {len(self.dependency_levels)}")
        print(f"Critical path length: {len(self.dependency_levels)}")
        print()

        for level_idx, level_nodes in enumerate(self.dependency_levels):
            print(
                f"Level {level_idx}: {len(level_nodes)} operations (can run in parallel)"
            )
            for node_id in level_nodes:
                node = self.ir_graph.node_map[node_id]
                deps = self.node_dependencies.get(node_id, set())
                print(f"  - {node.op_type} ({node.id}) [deps: {len(deps)}]")
            print()


def analyze_parallelism(ir_graph: IRGraph) -> ParallelExecutionPlan:
    """
    Convenience function to analyze an IR graph for parallelism.

    Usage:
        plan = analyze_parallelism(builder.ir_graph)
        if plan.use_parallel:
            # Use parallel execution
            json_payload = plan.to_json()
        else:
            # Fall back to sequential
            json_payload = plan.sequential_plan.to_json()
    """
    analyzer = ParallelAnalyzer(ir_graph)
    return analyzer.analyze()


def is_data_parallel_op(op_type: str) -> bool:
    """
    Check if an operation is data-parallel (can be parallelized element-wise).

    Data-parallel operations:
    - Operate independently on each element
    - No cross-element dependencies
    - Safe to split across threads
    """
    data_parallel_ops = {
        # Arithmetic
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "pow",
        "sqrt",
        "exp",
        "log",
        # Comparison
        "compare",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        # Conditional
        "where",
        "select",
        # Element-wise transformations
        "map",
        "transform",
        "element_wise",
        # Bitwise (if added)
        "and",
        "or",
        "xor",
        "not",
    }
    return op_type in data_parallel_ops


def is_thread_safe_op(op_type: str) -> bool:
    """
    Check if an operation is thread-safe (can run concurrently with other ops).

    Thread-safe operations:
    - Pure (no side effects)
    - No shared mutable state
    - Deterministic
    """
    # Most operations are thread-safe in a functional IR
    # Exceptions are operations with side effects
    unsafe_ops = {
        "store",
        "attach",
        "save",  # Attribute mutations
        "print",
        "debug",  # I/O operations
    }
    return op_type not in unsafe_ops
