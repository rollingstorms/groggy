"""
IR optimization passes for the builder DSL.

Implements dead code elimination, constant folding, and common subexpression
elimination to reduce IR size and improve execution efficiency.
"""
from typing import Set, Dict, List, Any, Optional
from .graph import IRGraph
from .nodes import AnyIRNode, IRDomain


class IROptimizer:
    """Applies optimization passes to IR graphs."""
    
    def __init__(self, ir_graph: IRGraph):
        self.ir = ir_graph
        self.modified = False
    
    def optimize(self, passes: Optional[List[str]] = None) -> bool:
        """
        Run optimization passes on the IR graph.
        
        Args:
            passes: List of pass names to run. If None, runs all passes.
                   Available: 'dce', 'constant_fold', 'cse'
        
        Returns:
            True if any modifications were made
        """
        if passes is None:
            passes = ['constant_fold', 'cse', 'dce']
        
        total_modified = False
        
        for pass_name in passes:
            if pass_name == 'dce':
                modified = self.dead_code_elimination()
            elif pass_name == 'constant_fold':
                modified = self.constant_folding()
            elif pass_name == 'cse':
                modified = self.common_subexpression_elimination()
            else:
                raise ValueError(f"Unknown optimization pass: {pass_name}")
            
            total_modified = total_modified or modified
        
        return total_modified
    
    def dead_code_elimination(self) -> bool:
        """
        Remove operations that don't contribute to outputs.
        
        An operation is dead if:
        1. It has no side effects (not output, attach, etc.)
        2. Its result is never used by any live operation
        
        Returns:
            True if any nodes were removed
        """
        live_nodes = self._mark_live_nodes()
        
        dead_nodes = []
        for node in self.ir.nodes:
            if node.id not in live_nodes:
                dead_nodes.append(node)
        
        if dead_nodes:
            for node in dead_nodes:
                self.ir.nodes.remove(node)
                del self.ir.node_map[node.id]
                if node.output in self.ir.var_defs:
                    del self.ir.var_defs[node.output]
            return True
        
        return False
    
    def _mark_live_nodes(self) -> Set[str]:
        """
        Mark all nodes that contribute to outputs or have side effects.
        Uses backward reachability from output nodes.
        """
        live = set()
        worklist = []
        
        # Start with nodes that have side effects (outputs, attachments)
        for node in self.ir.nodes:
            if self._has_side_effects(node):
                live.add(node.id)
                worklist.append(node)
        
        # Backward traversal to mark dependencies
        while worklist:
            current = worklist.pop()
            deps = self.ir.get_dependencies(current)
            for dep in deps:
                if dep.id not in live:
                    live.add(dep.id)
                    worklist.append(dep)
        
        return live
    
    def _has_side_effects(self, node: AnyIRNode) -> bool:
        """Check if a node has side effects (outputs, attachments, etc.)."""
        # Attr attach operations have side effects
        if node.domain == IRDomain.ATTR and node.op_type in ('attach', 'store'):
            return True
        # Control flow operations have side effects
        if node.domain == IRDomain.CONTROL:
            return True
        return False
    
    def constant_folding(self) -> bool:
        """
        Evaluate constant expressions at compile time.
        
        Folds operations like:
        - scalar + scalar -> scalar
        - scalar * scalar -> scalar
        - constant broadcasts
        
        Returns:
            True if any constants were folded
        """
        modified = False
        
        for node in list(self.ir.nodes):
            folded = self._try_fold_constant(node)
            if folded is not None:
                # Update node to be a constant
                node.op_type = 'constant'
                node.inputs = []
                node.metadata['value'] = folded
                modified = True
        
        return modified
    
    def _try_fold_constant(self, node: AnyIRNode) -> Optional[Any]:
        """Try to fold a node into a constant value."""
        # Only fold pure arithmetic on core domain
        if node.domain != IRDomain.CORE:
            return None
        
        if node.op_type not in {'add', 'mul', 'sub', 'div'}:
            return None
        
        # Check if all inputs are constants
        input_values = []
        for input_var in node.inputs:
            def_node = self.ir.get_defining_node(input_var)
            if def_node and def_node.op_type == 'constant':
                input_values.append(def_node.metadata.get('value'))
            else:
                return None  # Not all inputs are constant
        
        if len(input_values) < 2:
            return None
        
        # Perform the operation
        try:
            lhs, rhs = input_values[0], input_values[1]
            if node.op_type == 'add':
                return lhs + rhs
            elif node.op_type == 'mul':
                return lhs * rhs
            elif node.op_type == 'sub':
                return lhs - rhs
            elif node.op_type == 'div':
                if rhs == 0:
                    return None  # Don't fold division by zero
                return lhs / rhs
        except Exception:
            return None
        
        return None
    
    def common_subexpression_elimination(self) -> bool:
        """
        Eliminate redundant computations.
        
        If two operations compute the same result from the same inputs,
        reuse the first computation and eliminate the second.
        
        Returns:
            True if any duplicates were eliminated
        """
        modified = False
        
        # Map from operation signature to output variable
        seen_ops: Dict[str, str] = {}
        replacements: Dict[str, str] = {}  # old_var -> new_var
        
        # Process nodes in order
        for node in self.ir.nodes:
            # Skip nodes with side effects
            if self._has_side_effects(node):
                continue
            
            # Skip nodes without outputs
            if not node.output:
                continue
            
            # Create signature for this operation
            signature = self._operation_signature(node)
            
            if signature in seen_ops:
                # Found duplicate - mark for replacement
                original_var = seen_ops[signature]
                replacements[node.output] = original_var
                modified = True
            else:
                seen_ops[signature] = node.output
        
        # Apply replacements
        if replacements:
            self._apply_replacements(replacements)
        
        return modified
    
    def _operation_signature(self, node: AnyIRNode) -> str:
        """
        Create a unique signature for an operation.
        Two nodes with the same signature compute the same result.
        """
        # Include domain and operation type
        sig_parts = [f"{node.domain.value}.{node.op_type}"]
        
        # Include sorted inputs
        input_str = ','.join(sorted(node.inputs))
        sig_parts.append(input_str)
        
        # Include relevant metadata (skip output names)
        meta_parts = []
        for key in sorted(node.metadata.keys()):
            if key not in {'output', 'name', 'id', 'line'}:
                meta_parts.append(f"{key}={node.metadata[key]}")
        if meta_parts:
            sig_parts.append(','.join(meta_parts))
        
        return '::'.join(sig_parts)
    
    def _apply_replacements(self, replacements: Dict[str, str]):
        """Replace all uses of old variable names with new ones."""
        # Update all node inputs
        for node in self.ir.nodes:
            node.inputs = [replacements.get(inp, inp) for inp in node.inputs]
        
        # Update var_defs and var_uses
        for old_var, new_var in replacements.items():
            if old_var in self.ir.var_defs:
                del self.ir.var_defs[old_var]
            if old_var in self.ir.var_uses:
                # Redirect uses
                for use_node in self.ir.var_uses[old_var]:
                    self.ir.var_uses[new_var].append(use_node)
                del self.ir.var_uses[old_var]


def optimize_ir(ir_graph: IRGraph, passes: Optional[List[str]] = None, max_iterations: int = 3) -> IRGraph:
    """
    Apply optimization passes to an IR graph iteratively.
    
    Args:
        ir_graph: The IR graph to optimize
        passes: List of optimization passes to apply
        max_iterations: Maximum number of optimization iterations
    
    Returns:
        Optimized IR graph
    """
    optimizer = IROptimizer(ir_graph)
    
    for i in range(max_iterations):
        modified = optimizer.optimize(passes)
        if not modified:
            # Reached fixed point
            break
    
    return ir_graph
