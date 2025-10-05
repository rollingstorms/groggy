#!/usr/bin/env python3
import groggy as gr

g = gr.Graph()
g.add_node(name='A', dept='Eng')
g.add_node(name='B', dept='Eng')
g.add_node(name='C', dept='Sales')

g.add_edge(0, 1)

print(f"Original: {g.node_count()} nodes, {g.edge_count()} edges")

groups = g.nodes.group_by('dept')
print(f"Groups: {len(groups)}")

merged = groups.merge()
print(f"Merged: {merged.node_count()} nodes, {merged.edge_count()} edges")
print("SUCCESS!")
