"""
IR optimization passes for the builder DSL.

Implements dead code elimination, constant folding, common subexpression
elimination, and operation fusion to reduce IR size and improve execution efficiency.
"""
from typing import Set, Dict, List, Any, Optional, Tuple
from .graph import IRGraph
from .nodes import AnyIRNode, IRDomain, CoreIRNode


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
                   Available: 'dce', 'constant_fold', 'cse', 'fuse_arithmetic', 'fuse_neighbor'
        
        Returns:
            True if any modifications were made
        """
        if passes is None:
            passes = ['constant_fold', 'cse', 'fuse_arithmetic', 'fuse_neighbor', 'dce']
        
        total_modified = False
        
        for pass_name in passes:
            if pass_name == 'dce':
                modified = self.dead_code_elimination()
            elif pass_name == 'constant_fold':
                modified = self.constant_folding()
            elif pass_name == 'cse':
                modified = self.common_subexpression_elimination()
            elif pass_name == 'fuse_arithmetic':
                modified = self.fuse_arithmetic()
            elif pass_name == 'fuse_neighbor':
                modified = self.fuse_neighbor_operations()
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


    def fuse_arithmetic(self) -> bool:
        """
        Fuse chains of arithmetic operations into single operations.
        
        Detects patterns like:
        - (a * b) + c -> fused_arithmetic("axpy", a, b, c)
        - (a + b) * c -> fused_arithmetic("mul_add", a, b, c)
        - a / (b + epsilon) -> fused_arithmetic("safe_div", a, b, epsilon)
        
        Returns:
            True if any operations were fused
        """
        modified = False
        
        # Look for fusable arithmetic patterns
        for node in list(self.ir.nodes):
            if node.domain != IRDomain.CORE:
                continue
            
            # Pattern: (a op1 b) op2 c -> fused
            fused = self._try_fuse_binary_chain(node)
            if fused:
                modified = True
                continue
            
            # Pattern: where(mask, a op b, c) -> fused conditional
            if node.op_type == 'where':
                fused = self._try_fuse_conditional(node)
                if fused:
                    modified = True
        
        return modified
    
    def _try_fuse_binary_chain(self, node: AnyIRNode) -> bool:
        """Try to fuse a chain of binary operations."""
        if node.op_type not in {'add', 'mul', 'sub', 'div'}:
            return False
        
        # Check if we can fuse with an input operation
        # For now, detect AXPY pattern: (a * b) + c
        if node.op_type == 'add' and len(node.inputs) == 2:
            for i, inp_var in enumerate(node.inputs):
                inp_node = self.ir.get_defining_node(inp_var)
                if inp_node and inp_node.op_type == 'mul' and len(inp_node.inputs) == 2:
                    # Found (a * b) + c pattern
                    other_inp = node.inputs[1 - i]
                    
                    # Only fuse if mul result is only used here
                    uses = self.ir.var_uses.get(inp_var, [])
                    if len(uses) == 1:
                        # Create fused node
                        node.op_type = 'fused_axpy'
                        node.inputs = list(inp_node.inputs) + [other_inp]
                        node.metadata['pattern'] = 'axpy'
                        
                        # Remove the mul node (will be cleaned up by DCE)
                        return True
        
        return False
    
    def _try_fuse_conditional(self, node: AnyIRNode) -> bool:
        """Try to fuse arithmetic operations within conditional (where)."""
        if node.op_type != 'where' or len(node.inputs) < 3:
            return False
        
        # Check if true or false branches contain simple operations
        mask_var, true_var, false_var = node.inputs[0], node.inputs[1], node.inputs[2]
        
        # Pattern: where(mask, 0, a * b) -> fused_where_mul
        true_node = self.ir.get_defining_node(true_var)
        false_node = self.ir.get_defining_node(false_var)
        
        # Check for zero in one branch
        is_true_zero = true_node and true_node.op_type == 'constant' and true_node.metadata.get('value') == 0
        is_false_zero = false_node and false_node.op_type == 'constant' and false_node.metadata.get('value') == 0
        
        if is_false_zero and true_node and true_node.op_type in {'mul', 'div', 'add'}:
            # Fuse where(mask, a op b, 0) -> fused_where_op
            uses = self.ir.var_uses.get(true_var, [])
            if len(uses) == 1:
                node.op_type = f'fused_where_{true_node.op_type}'
                node.inputs = [mask_var] + list(true_node.inputs)
                node.metadata['pattern'] = f'where_{true_node.op_type}'
                return True
        
        return False
    
    def fuse_neighbor_operations(self) -> bool:
        """
        Fuse graph neighbor operations with pre/post arithmetic.
        
        Patterns:
        - transform(x) -> neighbor_agg -> fused_neighbor_agg(x, pre_transform)
        - neighbor_agg -> transform(y) -> fused_neighbor_agg(x, post_transform)
        - transform(x) -> neighbor_agg -> transform(y) -> fully fused
        
        Returns:
            True if any operations were fused
        """
        modified = False
        
        for node in list(self.ir.nodes):
            if node.domain != IRDomain.GRAPH or node.op_type != 'neighbor_agg':
                continue
            
            # Check for pre-transform pattern
            if len(node.inputs) > 0:
                inp_var = node.inputs[0]
                inp_node = self.ir.get_defining_node(inp_var)
                
                # Can we fuse the input operation?
                if inp_node and inp_node.domain == IRDomain.CORE:
                    if inp_node.op_type in {'mul', 'div', 'where', 'fused_where_mul'}:
                        uses = self.ir.var_uses.get(inp_var, [])
                        if len(uses) == 1:  # Only used by this neighbor_agg
                            # Fuse pre-transform
                            node.op_type = f'fused_neighbor_{inp_node.op_type}'
                            node.inputs = list(inp_node.inputs)
                            node.metadata['pre_op'] = inp_node.op_type
                            node.metadata['agg'] = node.metadata.get('agg', 'sum')
                            modified = True
            
            # Check for post-transform pattern (scan uses of this node)
            if node.output:
                uses = self.ir.var_uses.get(node.output, [])
                if len(uses) == 1:
                    use_node = uses[0]
                    if use_node.domain == IRDomain.CORE and use_node.op_type in {'mul', 'add'}:
                        # Fuse post-transform into the neighbor_agg
                        if 'post_op' not in node.metadata:  # Don't double-fuse
                            node.metadata['post_op'] = use_node.op_type
                            node.metadata['post_inputs'] = [inp for inp in use_node.inputs if inp != node.output]
                            # Update output to skip the intermediate operation
                            node.output = use_node.output
                            modified = True
        
        return modified


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
