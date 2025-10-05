#!/usr/bin/env python3
import groggy as gr
import time

gt = gr.from_csv(
    nodes_filepath='comprehensive_test_objects_20250929_163811.csv',
    edges_filepath='comprehensive_test_methods_20250929_163811.csv',
    node_id_column='object_name',
    source_id_column='object_name',
    target_id_column='result_type'
)

g = gt.to_graph()

print("Starting viz server with node_size='total_methods'...")
g.viz.show(layout='force_directed', node_size='total_methods')

time.sleep(5)
print("\nCheck http://127.0.0.1:8080/ to see if nodes have different sizes")
