# Meta API Graph Builder

## Overview

This directory contains the dynamic **Meta API Graph Builder**. It discovers Groggy objects and methods directly from the live API, resolves return types via safe dynamic calls, and exports a canonical meta-graph bundle.

## Core Concept

**Groggy analyzes Groggy**:
- Nodes represent object types and return types.
- Edges represent methods connecting objects to their return values.
- The graph is generated directly (no JSON inputs).

## Primary Script

- `meta_api_graph_builder.py` - Dynamic builder that discovers methods, tests return types, and exports the canonical bundle.

## Quick Start

```bash
python notes/meta_api_discovery/meta_api_graph_builder.py
```

Optional flags:
- `--skip-tests` to skip dynamic method calls
- `--output-dir` to change the bundle location

## Output

The builder writes a Groggy bundle (nodes/edges/metadata) and a summary JSON:

- `notes/meta_api_discovery/meta_api_graph_bundle/`
- `notes/meta_api_discovery/meta_api_graph_bundle/meta_api_graph_summary.json`
