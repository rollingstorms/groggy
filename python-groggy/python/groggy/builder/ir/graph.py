"""
IR Graph structure for algorithm representation.

Represents an algorithm as a directed acyclic graph (DAG) of operations,
enabling analysis and optimization passes.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .nodes import AnyIRNode, IRDomain, IRNode


class IRGraph:
    """
    Directed acyclic graph of IR nodes representing an algorithm.

    The graph tracks:
    - All operation nodes
    - Data dependencies between operations
    - Variable definitions and uses
    - Domain statistics for optimization

    Example:
        >>> graph = IRGraph()
        >>> node1 = CoreIRNode("n1", "mul", ["a", "b"], "c")
        >>> node2 = CoreIRNode("n2", "add", ["c", "d"], "e")
        >>> graph.add_node(node1)
        >>> graph.add_node(node2)
        >>> print(graph.pretty_print())
    """

    def __init__(self, name: str = "algorithm"):
        """
        Create a new IR graph.

        Args:
            name: Name of the algorithm this graph represents
        """
        self.name = name
        self.nodes: List[AnyIRNode] = []
        self.node_map: Dict[str, AnyIRNode] = {}  # id -> node
        self.var_defs: Dict[str, AnyIRNode] = {}  # var_name -> defining node
        self.var_uses: Dict[str, List[AnyIRNode]] = defaultdict(
            list
        )  # var_name -> using nodes
        self._node_counter = 0

    def add_node(self, node: AnyIRNode) -> None:
        """
        Add a node to the graph and update dependency tracking.

        Args:
            node: IR node to add
        """
        self.nodes.append(node)
        self.node_map[node.id] = node

        # Track variable definition
        if node.output:
            self.var_defs[node.output] = node

        # Track variable uses
        for input_var in node.inputs:
            self.var_uses[input_var].append(node)

    def get_node(self, node_id: str) -> Optional[AnyIRNode]:
        """Get node by ID."""
        return self.node_map.get(node_id)

    def get_defining_node(self, var_name: str) -> Optional[AnyIRNode]:
        """Get the node that defines a variable."""
        return self.var_defs.get(var_name)

    def get_using_nodes(self, var_name: str) -> List[AnyIRNode]:
        """Get all nodes that use a variable."""
        return self.var_uses.get(var_name, [])

    def get_dependencies(self, node: AnyIRNode) -> List[AnyIRNode]:
        """
        Get all nodes that this node depends on (inputs).

        Args:
            node: Node to find dependencies for

        Returns:
            List of nodes that produce the inputs this node consumes
        """
        deps = []
        for input_var in node.inputs:
            def_node = self.get_defining_node(input_var)
            if def_node:
                deps.append(def_node)
        return deps

    def get_dependents(self, node: AnyIRNode) -> List[AnyIRNode]:
        """
        Get all nodes that depend on this node (consumers).

        Args:
            node: Node to find dependents for

        Returns:
            List of nodes that consume this node's output
        """
        if node.output:
            return self.get_using_nodes(node.output)
        return []

    def topological_order(self) -> List[AnyIRNode]:
        """
        Return nodes in topological order (dependencies before dependents).

        Returns:
            List of nodes in execution order
        """
        # Already in topological order if added correctly
        # But verify with a proper topological sort
        visited = set()
        order = []

        def visit(node: AnyIRNode):
            if node.id in visited:
                return
            visited.add(node.id)

            # Visit dependencies first
            for dep in self.get_dependencies(node):
                visit(dep)

            order.append(node)

        # Visit all nodes
        for node in self.nodes:
            visit(node)

        return order

    def stats(self) -> Dict[str, int]:
        """
        Get statistics about the IR graph.

        Returns:
            Dictionary with counts by domain and operation type
        """
        stats = {
            "total_nodes": len(self.nodes),
            "total_variables": len(self.var_defs),
        }

        # Count by domain
        domain_counts = defaultdict(int)
        op_counts = defaultdict(int)

        for node in self.nodes:
            domain_counts[node.domain.value] += 1
            op_counts[f"{node.domain.value}.{node.op_type}"] += 1

        stats.update(
            {f"domain_{domain}": count for domain, count in domain_counts.items()}
        )

        stats["operation_types"] = dict(op_counts)

        return stats

    def to_dot(self) -> str:
        """
        Generate Graphviz DOT representation for visualization.

        Returns:
            DOT format string
        """
        lines = [
            f'digraph "{self.name}" {{',
            "  rankdir=TB;",
            "  node [shape=box, style=rounded];",
            "",
        ]

        # Color by domain
        domain_colors = {
            IRDomain.CORE: "lightblue",
            IRDomain.GRAPH: "lightgreen",
            IRDomain.ATTR: "lightyellow",
            IRDomain.CONTROL: "lightcoral",
        }

        # Add nodes
        for node in self.nodes:
            color = domain_colors.get(node.domain, "lightgray")
            label = f"{node.domain.value}.{node.op_type}"
            if node.output:
                label += f"\\n→ {node.output}"
            lines.append(
                f'  "{node.id}" [label="{label}", fillcolor={color}, style=filled];'
            )

        lines.append("")

        # Add edges (data dependencies)
        for node in self.nodes:
            for input_var in node.inputs:
                def_node = self.get_defining_node(input_var)
                if def_node:
                    lines.append(
                        f'  "{def_node.id}" -> "{node.id}" [label="{input_var}"];'
                    )

        lines.append("}")
        return "\n".join(lines)

    def pretty_print(self) -> str:
        """
        Generate human-readable text representation.

        Returns:
            Pretty-printed string
        """
        lines = [f"Algorithm: {self.name}", "=" * 60]

        # Print statistics
        stats = self.stats()
        lines.append(
            f"Nodes: {stats['total_nodes']}, Variables: {stats['total_variables']}"
        )
        lines.append("")

        # Print nodes in topological order
        for i, node in enumerate(self.topological_order()):
            lines.append(f"{i:3d}. {node}")
            if node.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in node.metadata.items())
                lines.append(f"      [{meta_str}]")

        return "\n".join(lines)

    def to_steps(self, expand_blocks: bool = False) -> List[Dict]:
        """
        Convert IR graph back to legacy step list format.

        Args:
            expand_blocks: If True, expand execution blocks into flat steps
                          (fallback mode for runtimes without block support)

        Returns:
            List of step dictionaries for FFI serialization
        """
        from .nodes import ExecutionBlockNode

        steps = []
        for node in self.topological_order():
            if expand_blocks and isinstance(node, ExecutionBlockNode):
                # Expand block into flat steps
                steps.extend(node.expand_to_steps())
            else:
                # Regular step conversion
                steps.append(node.to_step())

        return steps

    def to_json(self) -> Dict:
        """
        Serialize entire graph to JSON-compatible dictionary.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
            "stats": self.stats(),
        }

    def rebuild_var_tracking(self) -> None:
        """
        Rebuild var_defs and var_uses from current node graph.

        Call this after fusion passes that remove nodes to ensure
        variable tracking is consistent with the actual graph structure.
        """
        self.var_defs.clear()
        self.var_uses = defaultdict(list)

        for node in self.nodes:
            # Track what each variable is defined by
            if node.output:
                self.var_defs[node.output] = node

            # Track what nodes use each variable
            for inp in node.inputs:
                if isinstance(inp, str):  # Variable reference (not literal)
                    self.var_uses[inp].append(node)

    def clone(self) -> "IRGraph":
        """
        Create a deep copy of this graph.

        Returns:
            New IRGraph with the same structure
        """
        new_graph = IRGraph(self.name)
        for node in self.nodes:
            new_graph.add_node(node)
        return new_graph

    def rebuild_var_tracking(self) -> None:
        """
        Rebuild var_defs and var_uses from current node graph.

        Call this after any optimization pass that modifies graph structure
        (adds/removes/replaces nodes) to ensure variable tracking stays consistent.

        This is necessary because when nodes are removed or modified, the var_defs
        and var_uses dictionaries can contain stale references to deleted nodes.
        """
        # Clear existing tracking
        self.var_defs.clear()
        self.var_uses.clear()

        # Rescan all nodes to rebuild tracking
        for node in self.nodes:
            # Track what each variable is defined by
            if node.output:
                self.var_defs[node.output] = node

            # Track what nodes use each variable
            for inp in node.inputs:
                if isinstance(inp, str):  # Variable reference (not a constant)
                    self.var_uses[inp].append(node)

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """Compact representation."""
        return f"IRGraph(name='{self.name}', nodes={len(self.nodes)})"
