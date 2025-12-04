"""
Execution context managers for structured algorithm blocks.

Provides context managers like message_pass() that capture operations
and execute them with special semantics (in-place updates, ordering, etc.).
"""

from typing import Any, Dict, Optional, Set

from groggy.builder.ir.nodes import (AnyIRNode, ExecutionBlockNode, IRDomain,
                                     IRNode)
from groggy.builder.varhandle import VarHandle


class MessagePassContext:
    """
    Context manager for message-passing / Gauss-Seidel update blocks.

    When active, captures operations performed on variables and records them
    as part of a structured execution block that will be executed with
    in-place, ordered semantics.

    Example:
        >>> with builder.message_pass(target=labels, include_self=True, ordered=True) as mp:
        ...     msgs = mp.pull(labels)
        ...     update = builder.core.mode(msgs, tie_break="lowest")
        ...     mp.apply(update)

    Key methods:
        - pull(source): Collect neighbor values (wraps collect_neighbor_values)
        - apply(values): Write values back to target variable in-place
    """

    # Operations not allowed inside message-pass blocks
    # These require special handling or are fundamentally incompatible
    UNSUPPORTED_OPS: Set[str] = {
        "random_walk",  # Non-deterministic, incompatible with ordered semantics
        "attr.load",  # Attribute loading should happen outside block
        "attr.attach",  # Attribute writing should happen outside block
        "loop",  # Control flow inside blocks not yet supported
        "until_converged",  # Control flow inside blocks not yet supported
    }

    def __init__(
        self,
        builder,
        target: VarHandle,
        include_self: bool = True,
        ordered: bool = True,
        name: Optional[str] = None,
        **options,
    ):
        """
        Create a message-pass context.

        Args:
            builder: The AlgorithmBuilder instance
            target: Variable being updated in this block
            include_self: Whether to include self-loops in neighbor aggregation
            ordered: Whether to process nodes in order (Gauss-Seidel) or parallel (Jacobi)
            name: Optional name for this block (for debugging/profiling)
            **options: Additional options (tie_break, direction, etc.)
        """
        self.builder = builder
        self.target = target
        self.include_self = include_self
        self.ordered = ordered
        self.name = name or f"mp_{target.name if hasattr(target, 'name') else 'block'}"
        self.options = options

        # Track whether apply() was called
        self._applied = False

        # Create the block node
        block_id = f"block_{self.builder.ir_graph._node_counter}"
        self.builder.ir_graph._node_counter += 1

        self.block_node = ExecutionBlockNode(
            node_id=block_id,
            mode="message_pass",
            target=target.name if hasattr(target, "name") else str(target),
            include_self=include_self,
            ordered=ordered,
            name=self.name,
            **options,
        )

        # Stack of captured nodes (operations performed in the block)
        self._captured_nodes = []

        # Previous context state (for restoring)
        self._prev_context = None

    def __enter__(self):
        """Enter the context - start capturing operations."""
        # Check for nested contexts (disallow for now)
        prev = getattr(self.builder, "_active_exec_context", None)
        if prev is not None:
            raise RuntimeError(
                f"Nested execution contexts are not supported. "
                f"Already inside context '{prev.name}', cannot enter '{self.name}'. "
                f"Close the outer context before opening a new one."
            )

        # Save previous context and install ourselves
        self._prev_context = prev
        self.builder._active_exec_context = self

        # Mark the starting point in the IR graph for operation capture
        self._ir_start_index = len(self.builder.ir_graph.nodes)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context - finalize the block."""
        # Restore previous context
        self.builder._active_exec_context = self._prev_context

        # Validate that apply() was called
        if not self._applied:
            raise RuntimeError(
                f"MessagePassContext '{self.name}' exited without calling apply(). "
                "You must call mp.apply(values) at least once in the block."
            )

        # Remove captured nodes from the main IR graph – they now live inside the block.
        self.builder._remove_ir_nodes(self._captured_nodes)

        # Finalize and add block to IR graph
        if self.builder.ir_graph is not None:
            self.builder._add_ir_node(self.block_node)

        return False  # Don't suppress exceptions

    def pull(self, source: Optional[VarHandle] = None) -> VarHandle:
        """
        Collect neighbor values (wraps collect_neighbor_values).

        Args:
            source: Variable to pull from neighbors (defaults to self.target)

        Returns:
            VarHandle containing list-valued neighbor data
        """
        if source is None:
            source = self.target

        # Use the builder's graph_ops to create collect_neighbor_values
        result = self.builder.graph_ops.collect_neighbor_values(
            source, include_self=self.include_self
        )

        return result

    def apply(self, values: VarHandle) -> None:
        """
        Apply computed values back to the target variable.

        This marks the write-back point and can be called multiple times
        in a block for progressive refinement.

        Args:
            values: Computed values to write to target

        Raises:
            RuntimeError: If called outside context or with invalid values
        """
        # Validate we're inside the context
        if self.builder._active_exec_context != self:
            raise RuntimeError(
                f"Cannot call apply() outside execution context. "
                f"apply() must be called while the context is active."
            )

        # Validate values is a VarHandle (not a scalar)
        if not isinstance(values, VarHandle):
            raise TypeError(
                f"apply() requires a VarHandle, got {type(values).__name__}. "
                f"If you have a scalar value, wrap it with builder.constant() first."
            )

        self._applied = True

        # Capture operations performed since context entered
        # Get all nodes added to IR since we entered the context
        for node in self.builder.ir_graph.nodes[self._ir_start_index :]:
            if node not in self._captured_nodes:
                # Validate and capture
                self._validate_operation(node)
                self._captured_nodes.append(node)
                self.block_node.add_body_node(node)

        # Also capture the computation chain leading to this value
        value_name = values.name if hasattr(values, "name") else str(values)
        final_node = self.builder.ir_graph.get_defining_node(value_name)

        if final_node and final_node not in self._captured_nodes:
            self._validate_operation(final_node)
            self._captured_nodes.append(final_node)
            self.block_node.add_body_node(final_node)

    def capture_node(self, node: AnyIRNode) -> None:
        """
        Capture an operation performed inside the block.

        Called by the builder when an operation is performed while this
        context is active.
        """
        # Validate the operation is supported
        self._validate_operation(node)

        self._captured_nodes.append(node)
        self.block_node.add_body_node(node)

    def _validate_operation(self, node: AnyIRNode) -> None:
        """
        Validate that an operation is allowed inside this execution block.

        Args:
            node: IR node to validate

        Raises:
            RuntimeError: If the operation is not supported
        """
        # Check against unsupported operations
        op_full_name = f"{node.domain.value}.{node.op_type}"

        if node.op_type in self.UNSUPPORTED_OPS or op_full_name in self.UNSUPPORTED_OPS:
            raise RuntimeError(
                f"Operation '{op_full_name}' is not supported inside message-pass blocks. "
                f"Context: '{self.name}'. "
                f"This operation requires special handling or is incompatible with "
                f"ordered execution semantics."
            )

        # Warn about potentially problematic operations
        if node.domain == IRDomain.ATTR:
            import warnings

            warnings.warn(
                f"Attribute operation '{op_full_name}' inside message-pass block '{self.name}'. "
                f"This may not work as expected. Consider moving attribute access outside the block.",
                UserWarning,
            )

        # Control flow operations are not yet supported
        if node.domain == IRDomain.CONTROL:
            raise RuntimeError(
                f"Control flow operation '{op_full_name}' is not supported inside message-pass blocks. "
                f"Context: '{self.name}'. "
                f"Loops and conditionals must be outside the execution block."
            )

    @property
    def options_dict(self) -> Dict[str, Any]:
        """Get all options as a dictionary."""
        return {
            "include_self": self.include_self,
            "ordered": self.ordered,
            **self.options,
        }
