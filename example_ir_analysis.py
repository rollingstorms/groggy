"""
Example: Using IR Dataflow Analysis

Demonstrates how to analyze an algorithm IR to find optimization opportunities.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

from groggy.builder.ir import (
    IRGraph, CoreIRNode, GraphIRNode, AttrIRNode,
    analyze_dataflow
)


def build_pagerank_ir():
    """Build a simplified PageRank IR (one iteration)."""
    graph = IRGraph("pagerank_one_iter")
    
    # Initialize ranks
    n1 = AttrIRNode("n1", "load", [], "ranks", name="ranks", default=1.0)
    
    # Get node degrees
    n2 = GraphIRNode("n2", "degree", [], "degrees")
    
    # Compute contribution normalization: 1 / (degree + epsilon)
    n3 = CoreIRNode("n3", "recip", ["degrees"], "inv_deg", epsilon=1e-9)
    
    # Compute contributions: ranks / degrees
    n4 = CoreIRNode("n4", "mul", ["ranks", "inv_deg"], "contrib")
    
    # Aggregate contributions from neighbors
    n5 = GraphIRNode("n5", "neighbor_agg", ["contrib"], "neighbor_sum", agg="sum")
    
    # Apply damping: 0.85 * neighbor_sum
    n6 = CoreIRNode("n6", "mul", ["neighbor_sum"], "damped", b=0.85)
    
    # Add teleport: damped + 0.15 / N
    n7 = CoreIRNode("n7", "add", ["damped"], "new_ranks", b=0.15)
    
    # Normalize ranks
    n8 = CoreIRNode("n8", "normalize", ["new_ranks"], "normalized")
    
    # Attach as result
    n9 = AttrIRNode("n9", "attach", ["normalized"], None, name="pagerank")
    
    for node in [n1, n2, n3, n4, n5, n6, n7, n8, n9]:
        graph.add_node(node)
    
    return graph


def main():
    print("=" * 70)
    print("IR Dataflow Analysis Example: PageRank")
    print("=" * 70)
    print()
    
    # Build IR
    print("Building PageRank IR...")
    graph = build_pagerank_ir()
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Variables: {len(graph.var_defs)}")
    print()
    
    # Run analysis
    print("Running dataflow analysis...")
    analysis = analyze_dataflow(graph)
    print("  Analysis complete!")
    print()
    
    # Show what we discovered
    print("=" * 70)
    print("Analysis Results")
    print("=" * 70)
    print()
    
    # 1. Fusion Opportunities
    print("1. FUSION OPPORTUNITIES")
    print("-" * 70)
    print(f"   Found {len(analysis.fusion_chains)} fusable operation chains:")
    print()
    
    for i, chain in enumerate(analysis.fusion_chains, 1):
        print(f"   Chain {i}: {chain.pattern}")
        print(f"   - Length: {len(chain.nodes)} operations")
        print(f"   - Benefit: {chain.fusion_benefit:.2f} (higher is better)")
        print(f"   - Operations:")
        for node in chain.nodes:
            print(f"     • {node.op_type}({', '.join(node.inputs)}) → {node.output}")
        print()
        
        # Explain benefit
        saved_calls = len(chain.nodes) - 1
        print(f"   → Fusing this chain would eliminate {saved_calls} FFI call(s)")
        print()
    
    # 2. Critical Path
    print("2. CRITICAL PATH (Performance Bottleneck)")
    print("-" * 70)
    print(f"   Length: {len(analysis.critical_path)} operations")
    print(f"   This is the minimum execution time even with infinite parallelism.")
    print()
    print("   Path:")
    for i, node in enumerate(analysis.critical_path, 1):
        op_str = f"{node.op_type}({', '.join(node.inputs)})"
        if node.output:
            op_str += f" → {node.output}"
        print(f"   {i}. {op_str}")
    print()
    
    # 3. Dependencies
    print("3. DATA DEPENDENCIES")
    print("-" * 70)
    print(f"   RAW (Read-After-Write): {len(analysis.raw_deps)} variables")
    print(f"   WAR (Write-After-Read): {len(analysis.war_deps)} variables")
    print(f"   WAW (Write-After-Write): {len(analysis.waw_deps)} variables")
    print()
    
    # Show a few examples
    if analysis.raw_deps:
        print("   Example RAW dependencies (must preserve order):")
        for var, readers in list(analysis.raw_deps.items())[:3]:
            print(f"   - {var} written, then read by: {readers}")
        print()
    
    # 4. Dead Code
    print("4. DEAD CODE")
    print("-" * 70)
    if analysis.dead_vars:
        print(f"   Found {len(analysis.dead_vars)} unused variables:")
        for var in analysis.dead_vars:
            print(f"   - {var} (can be eliminated)")
        print()
    else:
        print("   ✓ No dead code found (all variables are used)")
        print()
    
    # 5. Optimization Recommendations
    print("=" * 70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    if analysis.fusion_chains:
        print("1. APPLY OPERATION FUSION")
        print(f"   - Fuse {len(analysis.fusion_chains)} operation chains")
        print(f"   - Expected reduction: {sum(len(c.nodes) - 1 for c in analysis.fusion_chains)} FFI calls")
        print(f"   - Estimated speedup: 1.5-2x for arithmetic operations")
        print()
    
    if len(analysis.critical_path) > 5:
        print("2. OPTIMIZE CRITICAL PATH")
        print(f"   - Focus optimization on the {len(analysis.critical_path)} critical operations")
        print("   - Consider parallelizing off-critical-path work")
        print()
    
    if not analysis.dead_vars and not analysis.fusion_chains:
        print("✓ Algorithm is already well-optimized!")
        print("  - No dead code")
        print("  - No obvious fusion opportunities")
        print()
    
    # 6. Summary Statistics
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  IR Size: {len(graph.nodes)} nodes, {len(graph.var_defs)} variables")
    print(f"  Fusion Opportunities: {len(analysis.fusion_chains)}")
    print(f"  Potential FFI Call Savings: {sum(len(c.nodes) - 1 for c in analysis.fusion_chains)}")
    print(f"  Critical Path Length: {len(analysis.critical_path)}")
    print(f"  Dead Variables: {len(analysis.dead_vars)}")
    print()
    print("=" * 70)
    
    # Visualize
    print()
    print("To visualize this IR graph:")
    print("  >>> graph.to_dot()")
    print("  Paste output at: https://dreampuf.github.io/GraphvizOnline/")
    print()


if __name__ == "__main__":
    main()
