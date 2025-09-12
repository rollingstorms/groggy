# Jupyter Notebook Embedding Guide

This guide shows you how to embed interactive streaming tables directly in Jupyter notebook cells instead of opening them in separate browser tabs.

## Quick Start

There are now **two ways** to use interactive tables in Jupyter:

### Method 1: Browser Tab (Original)
```python
import groggy

g = groggy.Graph()
g.add_node(1, name='Alice', age=25)
g.add_node(2, name='Bob', age=30)

# Opens in separate browser tab
url = g.nodes.table().interactive()
print(f"View at: {url}")
```

### Method 2: Embedded in Cell (New!)
```python
from IPython.display import HTML, display

# Generate embedded iframe HTML
iframe_html = g.nodes.table().interactive_embed()

# Display directly in the cell
display(HTML(iframe_html))
```

## Convenient Helper Functions

For even easier usage, we provide utility functions:

```python
from groggy.jupyter_utils import display_table, display_graph_tables

# Display a single table inline
display_table(g.nodes.table())

# Display both nodes and edges tables
display_graph_tables(g)

# Custom height
display_table(g.nodes.table(), height=800)
```

## All Table Types Support Embedding

All table types now support both `interactive()` and `interactive_embed()`:

```python
# Base tables
table = g.nodes.table()
table.interactive()         # Browser tab
table.interactive_embed()   # Embedded iframe

# Nodes tables  
nodes = g.nodes.table()
nodes.interactive()
nodes.interactive_embed()

# Edges tables
edges = g.edges.table()
edges.interactive()
edges.interactive_embed()

# From CSV
table = groggy.BaseTable.from_csv('data.csv')
table.interactive_embed()
```

## Complete Example

Here's a full example you can run in Jupyter:

```python
import groggy
from IPython.display import HTML, display, Markdown
from groggy.jupyter_utils import display_table

# Create a sample graph
g = groggy.Graph()
g.add_node(1, name='Alice', age=25, city='NYC')
g.add_node(2, name='Bob', age=30, city='LA')  
g.add_node(3, name='Charlie', age=35, city='Chicago')
g.add_edge(1, 2, weight=0.8, type='friend')
g.add_edge(2, 3, weight=0.6, type='colleague')

# Display nodes table embedded in cell
display(Markdown("## Nodes Table"))
display_table(g.nodes.table())

# Display edges table embedded in cell  
display(Markdown("## Edges Table"))
display_table(g.edges.table())
```

## Features of Embedded Tables

The embedded streaming tables have all the same features as the browser version:

- **Virtual scrolling** for large datasets (millions of rows)
- **Real-time WebSocket streaming** 
- **Interactive filtering** and sorting
- **Responsive design** that fits the cell width
- **600px default height** (customizable)

## Customization Options

```python
# Custom height
iframe_html = table.interactive_embed()
iframe_html = iframe_html.replace('height="600px"', 'height="800px"')
display(HTML(iframe_html))

# Or use the helper function
display_table(table, height=800)

# Get just the HTML without displaying
from groggy.jupyter_utils import embed_table_html
html = embed_table_html(table, height=400, width="80%")
```

## Fallback Behavior

The utility functions automatically fall back to browser mode if:
- Not running in Jupyter
- IPython is not available
- Any errors occur during embedding

This ensures your code works in all environments!

## Performance Notes

- Each embedded table starts its own WebSocket server on a random port
- The server runs in a background thread/tokio runtime
- Virtual scrolling means only visible rows are loaded
- LRU caching provides smooth scrolling performance
- No difference in performance between embedded and browser modes

Enjoy your embedded interactive tables! 🚀📊