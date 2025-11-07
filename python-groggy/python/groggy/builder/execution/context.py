"""
Execution context managers for structured algorithm blocks.

Provides context managers like message_pass() that capture operations
and execute them with special semantics (in-place updates, ordering, etc.).
"""

from typing import Optional, Any, Dict
from groggy.builder.varhandle import VarHandle
from groggy.builder.ir.nodes import ExecutionBlockNode, IRNode, AnyIRNode


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
    
    def __init__(self, builder, target: VarHandle, include_self: bool = True,
                 ordered: bool = True, name: Optional[str] = None, **options):
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
            target=target.name if hasattr(target, 'name') else str(target),
            include_self=include_self,
            ordered=ordered,
            name=self.name,
            **options
        )
        
        # Stack of captured nodes (operations performed in the block)
        self._captured_nodes = []
        
        # Previous context state (for restoring)
        self._prev_context = None
    
    def __enter__(self):
        """Enter the context - start capturing operations."""
        # Save previous context and install ourselves
        self._prev_context = getattr(self.builder, '_active_exec_context', None)
        self.builder._active_exec_context = self
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
        
        # Finalize and add block to IR graph
        if self.builder.ir_graph is not None:
            self.builder.ir_graph.add_node(self.block_node)
        
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
            source,
            include_self=self.include_self
        )
        
        return result
    
    def apply(self, values: VarHandle) -> None:
        """
        Apply computed values back to the target variable.
        
        This marks the write-back point and can be called multiple times
        in a block for progressive refinement.
        
        Args:
            values: Computed values to write to target
        """
        self._applied = True
        
        # Capture the computation chain leading to this value
        # Walk backwards from the final value to find all operations
        # that were performed inside this context
        value_name = values.name if hasattr(values, 'name') else str(values)
        
        # Find the node that produces this value
        final_node = self.builder.ir_graph.get_defining_node(value_name)
        if final_node:
            # For Phase 1: just capture the final node
            # Phase 2 would do a full dependency walk to capture the sub-DAG
            self.block_node.add_body_node(final_node)
            
            # Also capture any direct dependencies
            for dep in self.builder.ir_graph.get_dependencies(final_node):
                # Only capture nodes created during this context
                # (simple heuristic: check if they're recent)
                if dep not in self._captured_nodes:
                    self._captured_nodes.append(dep)
                    self.block_node.add_body_node(dep)
    
    def capture_node(self, node: AnyIRNode) -> None:
        """
        Capture an operation performed inside the block.
        
        Called by the builder when an operation is performed while this
        context is active.
        """
        self._captured_nodes.append(node)
        self.block_node.add_body_node(node)
    
    @property
    def options_dict(self) -> Dict[str, Any]:
        """Get all options as a dictionary."""
        return {
            "include_self": self.include_self,
            "ordered": self.ordered,
            **self.options
        }
