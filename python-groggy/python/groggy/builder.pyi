"""
Type stubs for groggy.builder module.

This file provides type hints for IDE autocomplete and type checking.
"""

from typing import Any, ContextManager, Dict, Optional, Union

from groggy.algorithms.base import AlgorithmHandle

class LoopContext:
    """Context manager for loop body."""

    builder: AlgorithmBuilder
    iterations: int
    start_step: Optional[int]
    loop_vars: Dict[str, VarHandle]

    def __init__(self, builder: AlgorithmBuilder, iterations: int) -> None: ...
    def __enter__(self) -> LoopContext: ...
    def __exit__(self, *args: Any) -> None: ...

class VarHandle:
    """
    Handle representing a variable in the algorithm builder.

    Variables track intermediate results as the algorithm is built.
    """

    name: str
    builder: AlgorithmBuilder

    def __init__(self, name: str, builder: AlgorithmBuilder) -> None: ...
    def __repr__(self) -> str: ...

class SubgraphHandle:
    """
    Handle representing a reference to the input subgraph.

    This allows algorithms to explicitly reference the input subgraph
    when needed, though most operations implicitly work on the input.
    """

    name: str
    builder: AlgorithmBuilder

    def __init__(self, name: str, builder: AlgorithmBuilder) -> None: ...
    def __repr__(self) -> str: ...

class CoreOps:
    """Namespace for core step primitives."""

    builder: AlgorithmBuilder

    def __init__(self, builder: AlgorithmBuilder) -> None: ...
    def add(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise addition or scalar addition.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result

        Example:
            >>> result = builder.core.add(values, 1.0)
            >>> combined = builder.core.add(values1, values2)
        """
        ...

    def sub(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise subtraction or scalar subtraction.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result
        """
        ...

    def mul(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise multiplication or scalar multiplication.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result

        Example:
            >>> scaled = builder.core.mul(values, 0.85)
        """
        ...

    def div(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise division or scalar division.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result
        """
        ...

    def update_in_place(
        self,
        source: VarHandle,
        target: VarHandle,
        ordered: bool = False,
    ) -> VarHandle:
        """
        Update a target map in-place with values from another map.
        """
        ...

    def neighbor_mode_update(
        self,
        target: VarHandle,
        include_self: bool = True,
        tie_break: str = "lowest",
        ordered: bool = True,
    ) -> VarHandle:
        """
        Update labels in-place by taking the mode of neighbor labels.
        """
        ...

    def normalize_sum(self, values: VarHandle) -> VarHandle:
        """
        Normalize values so they sum to 1.0.

        Args:
            values: Values to normalize

        Returns:
            VarHandle for normalized values

        Example:
            >>> normalized = builder.core.normalize_sum(ranks)
        """
        ...

class AlgorithmBuilder:
    """Builder for composing custom algorithms from step primitives."""

    name: str
    steps: list[Dict[str, Any]]
    variables: Dict[str, VarHandle]
    core: CoreOps

    def __init__(self, name: str) -> None:
        """
        Create a new algorithm builder.

        Args:
            name: Name for the algorithm
        """
        ...

    def auto_var(self, prefix: str = "var") -> VarHandle:
        """
        Create a unique variable name with the given prefix.

        This is the public interface to the variable name generator,
        useful when you need to create temporary variables.

        Args:
            prefix: Prefix for the variable name

        Returns:
            VarHandle with a unique name

        Example:
            >>> temp = builder.auto_var("temp")
            >>> print(temp.name)  # "temp_0"
        """
        ...

    def input(self, name: str = "subgraph") -> SubgraphHandle:
        """
        Create a reference to the input subgraph.

        This allows algorithms to explicitly reference the input subgraph,
        though most operations implicitly work on the input by default.

        Args:
            name: Name for the input reference (default: "subgraph")

        Returns:
            SubgraphHandle for the input

        Example:
            >>> sg = builder.input("graph")
            >>> # Most operations don't need explicit input reference
            >>> # but it's available for advanced use cases
        """
        ...

    def init_nodes(self, default: Any = 0.0) -> VarHandle:
        """
        Initialize node values.

        Args:
            default: Default value for all nodes

        Returns:
            Variable handle for initialized values
        """
        ...

    def load_attr(self, attr: str, default: Any = 0.0) -> VarHandle:
        """
        Load node attribute values into a variable.

        This allows algorithms to read existing node attributes from the graph.

        Args:
            attr: Name of the node attribute to load
            default: Default value for nodes without the attribute

        Returns:
            Variable handle containing the attribute values

        Example:
            >>> # Load existing node weights
            >>> weights = builder.load_attr("weight", default=1.0)
            >>> scaled = builder.core.mul(weights, 2.0)
        """
        ...

    def load_edge_attr(self, attr: str, default: Any = 0.0) -> VarHandle:
        """
        Load edge attribute values into a variable.

        This creates a mapping from (source, target) pairs to attribute values.

        Args:
            attr: Name of the edge attribute to load
            default: Default value for edges without the attribute

        Returns:
            Variable handle containing the edge attribute values

        Example:
            >>> # Load edge weights for weighted aggregation
            >>> edge_weights = builder.load_edge_attr("weight", default=1.0)
        """
        ...

    def node_degrees(self, nodes: VarHandle) -> VarHandle:
        """
        Compute node degrees.

        Args:
            nodes: Input node variable

        Returns:
            Variable handle for degrees
        """
        ...

    def normalize(self, values: VarHandle, method: str = "sum") -> VarHandle:
        """
        Normalize values.

        Args:
            values: Input values
            method: Normalization method ("sum", "max", "minmax")

        Returns:
            Variable handle for normalized values
        """
        ...

    def iterate(self, count: int) -> LoopContext:
        """
        Create a loop that repeats for `count` iterations.

        Steps added within the context will be repeated, with variables
        properly tracked across iterations.

        Args:
            count: Number of iterations

        Returns:
            Loop context manager

        Example:
            >>> with builder.iterate(20):
            ...     sums = builder.map_nodes(
            ...         "sum(ranks[neighbors(node)])",
            ...         inputs={"ranks": ranks}
            ...     )
            ...     ranks = builder.var("ranks", builder.core.mul(sums, 0.85))
        """
        ...

    def map_nodes(
        self, fn: str, inputs: Optional[Dict[str, VarHandle]] = None
    ) -> VarHandle:
        """
        Map expression over nodes with access to neighbors.

        This allows aggregating neighbor values or applying expressions to each node.

        Args:
            fn: Expression string (e.g., "sum(ranks[neighbors(node)])")
            inputs: Variable context for the expression

        Returns:
            VarHandle for the result

        Example:
            >>> # Sum neighbor values
            >>> sums = builder.map_nodes(
            ...     "sum(ranks[neighbors(node)])",
            ...     inputs={"ranks": ranks}
            ... )
            >>>
            >>> # Most common neighbor label
            >>> modes = builder.map_nodes(
            ...     "mode(labels[neighbors(node)])",
            ...     inputs={"labels": labels}
            ... )
        """
        ...

    def var(self, name: str, value: VarHandle) -> VarHandle:
        """
        Create or reassign a variable.

        This is useful for updating variables in loops, allowing you to
        reuse the same variable name across iterations.

        Args:
            name: Variable name
            value: Value to assign

        Returns:
            VarHandle with the given name

        Example:
            >>> ranks = builder.init_nodes(1.0)
            >>> with builder.iterate(10):
            >>>     updated = builder.core.mul(ranks, 0.85)
            >>>     ranks = builder.var("ranks", updated)
        """
        ...

    def attach_as(self, attr_name: str, values: VarHandle) -> None:
        """
        Attach values as a node attribute.

        Args:
            attr_name: Name for the output attribute
            values: Values to attach
        """
        ...

    def build(self, validate: bool = True) -> BuiltAlgorithm:
        """
        Build the algorithm from the composed steps.

        Args:
            validate: If True, validate pipeline before building (default: True)

        Returns:
            Built algorithm ready for use

        Raises:
            ValidationError: If validation fails and validate=True
        """
        ...

class BuiltAlgorithm(AlgorithmHandle):
    """Algorithm built from composed steps."""

    def __init__(self, name: str, steps: list[Dict[str, Any]]) -> None:
        """
        Create a built algorithm.

        Args:
            name: Algorithm name
            steps: List of step specifications
        """
        ...

    @property
    def id(self) -> str:
        """Get the algorithm identifier."""
        ...

    def to_spec(self) -> Dict[str, Any]:
        """Convert to a pipeline spec entry usable by the Rust executor."""
        ...

    def __repr__(self) -> str: ...

def builder(name: str) -> AlgorithmBuilder:
    """
    Create a new algorithm builder.

    Args:
        name: Name for the algorithm

    Returns:
        Algorithm builder instance

    Example:
        >>> from groggy import builder
        >>> b = builder("my_algo")
        >>> nodes = b.init_nodes()
        >>> degrees = b.node_degrees(nodes)
        >>> b.attach_as("degree", degrees)
        >>> algo = b.build()
    """
    ...
