# Groggy Python: analysis.py

Provides functions for change tracking, provenance, and diffing between graph states or entities.

- **show_changes, show_entity_changes, track_attribute_changes**: Provenance and audit utilities.

## Example
```python
from groggy.analysis import show_changes
changes = show_changes(G, 'experiment-1')
```
