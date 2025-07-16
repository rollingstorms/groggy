# Groggy Python: graph/ Module

Contains submodules for advanced graph state, branching, and subgraph management.

- **state.py**: StateManager for saving/loading/branching graph states.
- **subgraph.py**: Subgraph class for filtered graph views with provenance.

## Example: State & Branching
```python
from groggy.graph.state import StateManager
sm = StateManager()
sm.save(G, "Initial commit")
sm.create_branch(G, "experiment-1")
sm.switch_branch(G, "experiment-1")
```

## Example: Subgraph
```python
from groggy.graph.subgraph import Subgraph
subG = Subgraph(G, filter_criteria={"type": "A"}, metadata={"created_by": "agent"})
print(subG.get_metadata())
```
