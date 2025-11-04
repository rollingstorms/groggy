"""Dump the PageRank spec to see what's being generated."""
from groggy.builder import AlgorithmBuilder
import json

def build_pr_small(n=10):
    builder = AlgorithmBuilder("test_pr")
    ranks = builder.init_nodes(default=1.0 / n)
    ranks = builder.var("ranks", ranks)
    degrees = builder.node_degrees(ranks)
    inv_degrees = builder.core.recip(degrees, epsilon=1e-12)
    is_sink = builder.core.compare(degrees, "eq", 0.0)
    
    with builder.iterate(3):  # Just 3 iterations for readable output
        contrib = builder.core.mul(ranks, inv_degrees)
        contrib = builder.core.where(is_sink, 0.0, contrib)
        neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        sink_contrib = builder.core.broadcast_scalar(sink_mass, ranks)
        damped_neighbors = builder.core.mul(neighbor_sum, 0.85)
        damped_sinks = builder.core.mul(sink_contrib, 0.85 / n)
        teleport = (1.0 - 0.85) / n
        ranks = builder.core.add(damped_neighbors, damped_sinks)
        ranks = builder.core.add(ranks, teleport)
        ranks = builder.var("ranks", ranks)
    
    builder.attach_as("pagerank", ranks)
    algo = builder.build()
    return algo

algo = build_pr_small()
spec = algo.to_spec()

print(f"Algorithm: {spec['name']}")
print(f"Steps: {len(spec['steps'])}\n")

# Look for alias steps with 'ranks' - these show the variable redefinitions
print("Alias steps involving 'ranks':")
for i, step in enumerate(spec['steps']):
    if step.get('id') == 'alias':
        source = step.get('source', '?')
        target = step.get('target', '?')
        if 'ranks' in source or 'ranks' in target:
            print(f"  Step {i}: {source} -> {target}")

# Show the final attachment
print("\nFinal attachment steps:")
for i, step in enumerate(spec['steps']):
    if step.get('id') == 'attach':
        print(f"  Step {i}: attach {step.get('source', '?')} as '{step.get('name', '?')}'")

# Show a sample iteration - find where iter0 and iter1 start
print("\n\nSample: Steps that output to *_iter0 (first iteration):")
for i, step in enumerate(spec['steps']):
    output = step.get('output', '')
    if 'iter0' in output:
        step_id = step.get('id', '?')
        if step_id in ['core.mul', 'core.add']:
            left = step.get('left', '?')
            right = step.get('right', '?')
            print(f"  {i}: {step_id} {left}, {right} -> {output}")
        elif step_id == 'core.neighbor_agg':
            inp = step.get('input', '?')
            print(f"  {i}: neighbor_agg {inp} -> {output}")
        elif step_id == 'core.where':
            print(f"  {i}: where cond={step.get('condition')}, true={step.get('if_true')}, false={step.get('if_false')} -> {output}")
        else:
            print(f"  {i}: {step_id} -> {output}")

print("\n\nSample: Steps that output to *_iter1 (second iteration):")
for i, step in enumerate(spec['steps']):
    output = step.get('output', '')
    if 'iter1' in output:
        step_id = step.get('id', '?')
        if step_id in ['core.mul', 'core.add']:
            left = step.get('left', '?')
            right = step.get('right', '?')
            print(f"  {i}: {step_id} {left}, {right} -> {output}")
        elif step_id == 'core.neighbor_agg':
            inp = step.get('input', '?')
            print(f"  {i}: neighbor_agg {inp} -> {output}")
        else:
            print(f"  {i}: {step_id} -> {output}")
