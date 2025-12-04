"""
Variable handles with operator overloading for natural DSL syntax.

This module provides VarHandle, GraphHandle, and SubgraphHandle classes
that enable mathematical expressions in algorithm definitions.
"""

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from groggy.builder.algorithm_builder import AlgorithmBuilder


class VarHandle:
    """
    Handle representing a variable with operator overloading.

    Variables track intermediate results as the algorithm is built.
    Operators map to builder operations for natural mathematical syntax.

    Example:
        >>> a = builder._new_var("a")
        >>> b = builder._new_var("b")
        >>> c = a + b * 2.0  # Natural syntax
        >>> mask = a > 0.5
        >>> result = mask.where(a, 0.0)
    """

    def __init__(self, name: str, builder: "AlgorithmBuilder"):
        """
        Create a variable handle.

        Args:
            name: Variable name
            builder: Parent builder
        """
        self.name = name
        self.builder = builder

    # Arithmetic operators
    def __add__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Addition: a + b"""
        return self.builder.core.add(self, other)

    def __radd__(self, other: Union[float, int]) -> "VarHandle":
        """Reverse addition: 2.0 + a"""
        return self.builder.core.add(other, self)

    def __sub__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Subtraction: a - b"""
        return self.builder.core.sub(self, other)

    def __rsub__(self, other: Union[float, int]) -> "VarHandle":
        """Reverse subtraction: 2.0 - a"""
        return self.builder.core.sub(other, self)

    def __mul__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Multiplication: a * b"""
        return self.builder.core.mul(self, other)

    def __rmul__(self, other: Union[float, int]) -> "VarHandle":
        """Reverse multiplication: 2.0 * a"""
        return self.builder.core.mul(other, self)

    def __truediv__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Division: a / b"""
        return self.builder.core.div(self, other)

    def __rtruediv__(self, other: Union[float, int]) -> "VarHandle":
        """Reverse division: 2.0 / a"""
        return self.builder.core.div(other, self)

    def __neg__(self) -> "VarHandle":
        """Negation: -a"""
        return self.builder.core.mul(self, -1.0)

    def __pow__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Power: a ** b"""
        # Will be implemented in Phase 2 when we add pow to CoreOps
        raise NotImplementedError("Power operator not yet implemented")

    # Comparison operators (return mask VarHandles with 0.0/1.0 values)
    def __eq__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Equality: a == b (returns mask)"""
        return self.builder.core.compare(self, "eq", other)

    def __ne__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Inequality: a != b (returns mask)"""
        return self.builder.core.compare(self, "ne", other)

    def __lt__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Less than: a < b (returns mask)"""
        return self.builder.core.compare(self, "lt", other)

    def __le__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Less than or equal: a <= b (returns mask)"""
        return self.builder.core.compare(self, "le", other)

    def __gt__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Greater than: a > b (returns mask)"""
        return self.builder.core.compare(self, "gt", other)

    def __ge__(self, other: Union["VarHandle", float, int]) -> "VarHandle":
        """Greater than or equal: a >= b (returns mask)"""
        return self.builder.core.compare(self, "ge", other)

    # Logical operators (for masks)
    def __invert__(self) -> "VarHandle":
        """Logical NOT: ~mask (inverts 0.0 <-> 1.0)"""
        return self.builder.core.compare(self, "eq", 0.0)

    def __and__(self, other: "VarHandle") -> "VarHandle":
        """Logical AND: mask1 & mask2 (element-wise multiplication)"""
        return self.builder.core.mul(self, other)

    def __or__(self, other: "VarHandle") -> "VarHandle":
        """Logical OR: mask1 | mask2 (element-wise max)"""
        # Will be implemented when we add max to CoreOps in Phase 2
        raise NotImplementedError(
            "Logical OR not yet implemented (requires max operator)"
        )

    # Matrix/graph operator
    def __matmul__(self, other: "VarHandle") -> "VarHandle":
        """
        Neighbor aggregation operator: G @ values

        Note: This is called when VarHandle is on the left side.
        GraphHandle also implements __matmul__ for G @ values syntax.
        """
        # This will be connected to GraphOps in Phase 2
        return self.builder.core.neighbor_agg(other, "sum")

    # Fluent methods
    def where(
        self,
        if_true: Union["VarHandle", float],
        if_false: Union["VarHandle", float] = 0.0,
    ) -> "VarHandle":
        """
        Conditional selection: mask.where(true_vals, false_vals)

        Args:
            if_true: Value to use where self is non-zero
            if_false: Value to use where self is zero (default: 0.0)

        Returns:
            VarHandle with selected values

        Example:
            >>> is_sink = (degrees == 0.0)
            >>> contrib = is_sink.where(0.0, ranks * inv_degrees)
        """
        return self.builder.core.where(self, if_true, if_false)

    def reduce(self, op: str = "sum") -> "VarHandle":
        """
        Reduce to scalar: values.reduce("sum")

        Args:
            op: Reduction operation ('sum', 'mean', 'min', 'max')

        Returns:
            Scalar VarHandle

        Example:
            >>> total = values.reduce("sum")
            >>> avg = values.reduce("mean")
        """
        return self.builder.core.reduce_scalar(self, op)

    def degrees(self) -> "VarHandle":
        """
        Get node degrees: nodes.degrees()

        Returns:
            VarHandle with degree for each node

        Example:
            >>> ranks = G.nodes(1.0)
            >>> deg = ranks.degrees()
        """
        return self.builder.node_degrees(self)

    def normalize(self, method: str = "sum") -> "VarHandle":
        """
        Normalize values: values.normalize()

        Args:
            method: Normalization method ('sum', 'max', 'minmax')

        Returns:
            VarHandle with normalized values

        Example:
            >>> ranks = ranks.normalize()
        """
        return self.builder.core.normalize_sum(self)

    def __repr__(self) -> str:
        return f"Var({self.name})"


class GraphHandle:
    """
    Handle representing the input graph with topological operations.

    Provides convenient methods for graph-level operations and initialization.

    Example:
        >>> G = builder.graph()
        >>> ranks = G.nodes(1.0 / G.N)
        >>> neighbor_sum = G @ ranks
        >>> n = G.N  # Number of nodes
    """

    def __init__(self, builder: "AlgorithmBuilder"):
        """
        Create a graph handle.

        Args:
            builder: Parent algorithm builder
        """
        self.builder = builder

    def nodes(self, default: float = 0.0, unique: bool = False) -> VarHandle:
        """
        Initialize node values: G.nodes(1.0)

        Args:
            default: Default value for all nodes (ignored if unique=True)
            unique: If True, initialize with sequential indices (0, 1, 2, ...)

        Returns:
            VarHandle with initialized node values

        Example:
            >>> ranks = G.nodes(1.0)
            >>> labels = G.nodes(unique=True)  # For LPA
        """
        return self.builder.init_nodes(default=default, unique=unique)

    def edges(self, default: float = 0.0) -> VarHandle:
        """
        Initialize edge values: G.edges(1.0)

        Args:
            default: Default value for all edges

        Returns:
            VarHandle with initialized edge values
        """
        # This will be implemented when we add edge support
        raise NotImplementedError("Edge initialization not yet implemented")

    def var(self, name: str, value: Union[VarHandle, float, int]) -> VarHandle:
        """
        Create a logical variable for loop updates.

        Convenience method that delegates to builder.var().
        Useful inside iterate() blocks.

        Args:
            name: Variable name for the loop
            value: Initial or updated value

        Returns:
            VarHandle representing the variable

        Example:
            >>> with sG.iterate(100):
            >>>     new_ranks = 0.85 * neighbor_sum + 0.15
            >>>     ranks = sG.var("ranks", new_ranks)
        """
        return self.builder.var(name, value)

    def iterate(self, count: int):
        """
        Fixed iteration loop - cleaner than sG.builder.iter.loop().

        Args:
            count: Number of iterations

        Returns:
            Context manager for loop body

        Example:
            >>> with sG.iterate(100):
            >>>     neighbor_sum = sG @ ranks
            >>>     ranks = sG.var("ranks", 0.85 * neighbor_sum + 0.15)
        """
        return self.builder.iterate(count)

    def until_converged(
        self, watched: VarHandle, tol: float = 1e-6, max_iter: int = 1000
    ):
        """
        Loop until convergence (future feature).

        Args:
            watched: Variable to watch for convergence
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Example:
            >>> with sG.until_converged(ranks, tol=1e-6):
            >>>     ranks = sG.var("ranks", update_ranks(ranks))
        """
        return self.builder.iter.until_converged(watched, tol, max_iter)

    def __matmul__(self, values: VarHandle) -> VarHandle:
        """
        Neighbor aggregation: G @ values

        Args:
            values: Node values to aggregate

        Returns:
            VarHandle with aggregated neighbor values

        Example:
            >>> neighbor_sum = G @ ranks
        """
        return self.builder.core.neighbor_agg(values, "sum")

    @property
    def N(self) -> VarHandle:
        """
        Node count scalar: G.N

        Returns:
            Scalar VarHandle containing number of nodes

        Example:
            >>> ranks = G.nodes(1.0 / G.N)
        """
        return self.builder.graph_node_count()

    @property
    def M(self) -> VarHandle:
        """
        Edge count scalar: G.M

        Returns:
            Scalar VarHandle containing number of edges

        Example:
            >>> density = G.M / (G.N * G.N)
        """
        return self.builder.graph_edge_count()

    def __repr__(self) -> str:
        return "Graph"


class SubgraphHandle:
    """
    Handle representing a reference to the input subgraph.

    This allows algorithms to explicitly reference the input subgraph
    when needed, though most operations implicitly work on the input.
    """

    def __init__(self, name: str, builder: "AlgorithmBuilder"):
        """
        Create a subgraph handle.

        Args:
            name: Reference name for the subgraph
            builder: Parent builder
        """
        self.name = name
        self.builder = builder

    def __repr__(self) -> str:
        return f"SubgraphHandle('{self.name}')"
