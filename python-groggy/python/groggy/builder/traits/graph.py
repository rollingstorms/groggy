"""
Graph topology and structural operations.

This module provides GraphOps, containing operations that work with
graph structure: degrees, neighbor aggregation, subgraphs, etc.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from groggy.builder.algorithm_builder import AlgorithmBuilder

from groggy.builder.varhandle import VarHandle


class GraphOps:
    """Graph topology and structural operations."""

    def __init__(self, builder: "AlgorithmBuilder"):
        """
        Initialize graph operations.

        Args:
            builder: Parent algorithm builder
        """
        self.builder = builder

    def degree(self, nodes: Optional[VarHandle] = None) -> VarHandle:
        """
        Compute node degrees.

        Args:
            nodes: Optional node variable for context (if None, computes for all nodes)

        Returns:
            VarHandle with degree for each node

        Example:
            >>> # Get degrees for all nodes
            >>> degrees = builder.graph.degree()
            >>>
            >>> # Or via VarHandle fluent method
            >>> nodes = builder.init_nodes(1.0)
            >>> degrees = nodes.degrees()
        """
        var = self.builder._new_var("degrees")

        # For IR mode, we don't need a dummy source
        source_name = nodes.name if nodes else None

        if not source_name:
            # If no source provided, create a dummy one
            dummy = self.builder.init_nodes(0.0)
            source_name = dummy.name

        if self.builder.use_ir and self.builder.ir_graph is not None:
            from groggy.builder.ir.nodes import GraphIRNode, IRDomain

            node = GraphIRNode(
                node_id=f"node_{len(self.builder.ir_graph.nodes)}",
                op_type="degree",
                inputs=[source_name] if source_name else [],
                output=var.name,
            )
            self.builder._add_ir_node(node)
        else:
            step = {"type": "node_degree", "output": var.name, "source": source_name}
            self.builder.steps.append(step)

        return var

    def node_degree(self, nodes: Optional[VarHandle] = None) -> VarHandle:
        """
        Backward-compatible alias for degree().
        """
        return self.degree(nodes)

    def neighbor_agg(
        self, values: VarHandle, agg: str = "sum", weights: Optional[VarHandle] = None
    ) -> VarHandle:
        """
        Aggregate neighbor values for each node.

        For each node, aggregates values from its neighbors. Optionally
        applies weights to neighbor contributions.

        Args:
            values: Node values to aggregate
            agg: Aggregation type: 'sum', 'mean', 'min', 'max', 'mode'
            weights: Optional weights to apply to neighbor values

        Returns:
            VarHandle for aggregated neighbor values

        Note: Also accessible via G @ values for sum aggregation

        Example:
            >>> # PageRank: sum neighbor contributions
            >>> neighbor_sum = builder.graph.neighbor_agg(contrib, "sum")
            >>>
            >>> # Or using matrix operator
            >>> neighbor_sum = G @ contrib
            >>>
            >>> # With weights
            >>> weighted = builder.graph.neighbor_agg(ranks, "sum", weights=edge_weights)
        """
        var = self.builder._new_var("neighbor_agg")

        if self.builder.use_ir and self.builder.ir_graph is not None:
            from groggy.builder.ir.nodes import GraphIRNode, IRDomain

            inputs = [values.name]
            if weights is not None:
                inputs.append(weights.name)

            node = GraphIRNode(
                node_id=f"node_{len(self.builder.ir_graph.nodes)}",
                op_type="neighbor_agg",
                inputs=inputs,
                output=var.name,
                agg=agg,
            )
            self.builder._add_ir_node(node)
        else:
            step = {
                "type": "core.neighbor_agg",
                "source": values.name,
                "agg": agg,
                "output": var.name,
            }
            if weights is not None:
                step["weights"] = weights.name
            self.builder.steps.append(step)

        return var

    def collect_neighbor_values(
        self, values: VarHandle, include_self: bool = True
    ) -> VarHandle:
        """
        Collect neighbor values into lists for each node.

        For each node, gathers all neighbor values into a list. Used for
        algorithms like LPA that need to find the most common neighbor value.

        Args:
            values: Node values to collect
            include_self: Whether to include the node's own value in the list

        Returns:
            VarHandle for lists of neighbor values

        Example:
            >>> # LPA: collect neighbor labels (including self)
            >>> labels = builder.init_nodes(unique=True)
            >>> neighbor_labels = builder.graph.collect_neighbor_values(labels, include_self=True)
            >>> most_common = builder.core.mode(neighbor_labels)
        """
        var = self.builder._new_var("neighbor_values")

        if self.builder.use_ir and self.builder.ir_graph is not None:
            from groggy.builder.ir.nodes import CoreIRNode

            node = CoreIRNode(
                node_id=f"node_{len(self.builder.ir_graph.nodes)}",
                op_type="collect_neighbor_values",
                inputs=[values.name],
                output=var.name,
                include_self=include_self,
            )
            self.builder._add_ir_node(node)
        else:
            self.builder.steps.append(
                {
                    "type": "core.collect_neighbor_values",  # Keep same step type
                    "source": values.name,
                    "include_self": include_self,
                    "output": var.name,
                }
            )
        return var

    def neighbor_mode_update(
        self,
        target: VarHandle,
        include_self: bool = True,
        tie_break: str = "lowest",
        ordered: bool = True,
    ) -> VarHandle:
        """
        Update labels in-place by taking the mode of neighbor labels.

        This mirrors asynchronous LPA semantics where nodes update sequentially
        using the most common neighbor label (with optional self inclusion) and
        deterministic ordering.

        Args:
            target: Label map to update in place
            include_self: Whether to include the node's current label in the vote
            tie_break: 'lowest', 'highest', or 'keep' for first occurrence
            ordered: Process nodes in sorted order if True (recommended for determinism)

        Returns:
            VarHandle for updated labels (same as target)

        Example:
            >>> # Asynchronous LPA
            >>> labels = builder.init_nodes(unique=True)
            >>> with builder.iterate(10):
            ...     labels = builder.graph.neighbor_mode_update(
            ...         labels,
            ...         include_self=True,
            ...         ordered=True
            ...     )
        """
        self.builder.steps.append(
            {
                "type": "core.neighbor_mode_update",  # Keep same step type
                "target": target.name,
                "include_self": include_self,
                "tie_break": tie_break,
                "ordered": ordered,
                "output": target.name,
            }
        )
        return target

    def neighbors(self, nodes: VarHandle) -> VarHandle:
        """
        Get neighbor lists for each node (future feature).

        Returns:
            VarHandle containing lists of neighbor IDs

        Example:
            >>> # Get neighbors for all nodes
            >>> nodes = builder.init_nodes(0.0)
            >>> neighbor_lists = builder.graph.neighbors(nodes)
        """
        # This is a placeholder for future implementation
        raise NotImplementedError("neighbors() not yet implemented")

    def subgraph(self, node_mask: VarHandle) -> VarHandle:
        """
        Create induced subgraph from node mask (future feature).

        Args:
            node_mask: Binary mask (1.0 = include, 0.0 = exclude)

        Returns:
            Handle to induced subgraph

        Example:
            >>> # Create subgraph of high-degree nodes
            >>> degrees = builder.graph.degree()
            >>> high_deg = degrees > 10.0
            >>> sub = builder.graph.subgraph(high_deg)
        """
        # This is a placeholder for future implementation
        raise NotImplementedError("subgraph() not yet implemented")

    def connected_components(self) -> VarHandle:
        """
        Find connected components (future feature).

        Returns:
            Component label for each node

        Example:
            >>> # Find components
            >>> components = builder.graph.connected_components()
        """
        # This is a placeholder for future implementation
        raise NotImplementedError("connected_components() not yet implemented")

    def shortest_paths(
        self, sources: VarHandle, weights: Optional[VarHandle] = None
    ) -> VarHandle:
        """
        Compute shortest paths from sources (future feature).

        Args:
            sources: Binary mask of source nodes
            weights: Optional edge weights

        Returns:
            Distance from nearest source for each node

        Example:
            >>> # Single-source shortest paths
            >>> source_mask = (node_ids == 0)
            >>> distances = builder.graph.shortest_paths(source_mask)
        """
        # This is a placeholder for future implementation
        raise NotImplementedError("shortest_paths() not yet implemented")
