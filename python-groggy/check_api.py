#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy

g = groggy.Graph()
print("Available methods on Graph:")
print([m for m in dir(g) if not m.startswith('_')])