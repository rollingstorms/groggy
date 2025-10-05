#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr
import requests
import json
import time

# Quick test for the snapshot endpoint
g = gr.Graph()
g.add_node(label="Test", age=25, coords=[1.0, 2.0, 3.0])
g.add_node(label="Test2", tags=["a", "b"])

viz = g.viz
server = viz.server()
time.sleep(2)

try:
    response = requests.get("http://localhost:8083/debug/snapshot", timeout=5)
    if response.status_code == 200:
        data = response.json()
        if data.get('nodes'):
            first_node = data['nodes'][0]
            print(f"Node {first_node['id']} attributes:")
            for key, value in first_node['attributes'].items():
                print(f"  {key}: {value} (type: {type(value).__name__})")
        else:
            print("No nodes in snapshot")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")