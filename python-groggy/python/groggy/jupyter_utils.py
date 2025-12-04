"""
Jupyter notebook utilities for groggy

Provides convenient functions for displaying interactive tables in Jupyter notebooks.
"""


def display_table(table, height=600):
    """
    Display an interactive streaming table embedded in a Jupyter notebook cell.

    This function automatically handles the iframe generation and display,
    making it easy to visualize tables inline without opening separate browser tabs.

    Args:
        table: A groggy table object (BaseTable, NodesTable, or EdgesTable)
        height: Height of the iframe in pixels (default: 600)

    Example:
        ```python
        import groggy
        from groggy.jupyter_utils import display_table

        g = groggy.Graph()
        g.add_node(1, name='Alice', age=25)
        g.add_node(2, name='Bob', age=30)

        # Display nodes table inline
        display_table(g.nodes.table())
        ```
    """
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        # Check if we're in a Jupyter environment
        if get_ipython() is None:
            print("⚠️  Not running in Jupyter - falling back to browser mode")
            return table.interactive()

        # Generate embedded iframe HTML
        iframe_html = table.interactive_embed()

        # Apply custom height if specified
        if height != 600:
            iframe_html = iframe_html.replace('height="600px"', f'height="{height}px"')

        # Display the iframe
        display(HTML(iframe_html))

        print("📊 Interactive table embedded successfully!")
        return None

    except ImportError:
        print("⚠️  IPython not available - falling back to browser mode")
        return table.interactive()
    except Exception as e:
        print(f"❌ Error displaying embedded table: {e}")
        print("🔄 Falling back to browser mode...")
        return table.interactive()


def display_graph_tables(graph, show_nodes=True, show_edges=True, height=400):
    """
    Display both nodes and edges tables from a graph in separate embedded iframes.

    Args:
        graph: A groggy Graph object
        show_nodes: Whether to display the nodes table (default: True)
        show_edges: Whether to display the edges table (default: True)
        height: Height of each iframe in pixels (default: 400)

    Example:
        ```python
        import groggy
        from groggy.jupyter_utils import display_graph_tables

        g = groggy.Graph()
        g.add_node(1, name='Alice')
        g.add_node(2, name='Bob')
        g.add_edge(1, 2, weight=0.8)

        # Display both tables
        display_graph_tables(g)
        ```
    """
    try:
        from IPython import get_ipython
        from IPython.display import HTML, Markdown, display

        if get_ipython() is None:
            print("⚠️  Not running in Jupyter - falling back to browser mode")
            if show_nodes:
                print("📊 Nodes table:", graph.nodes.table().interactive())
            if show_edges:
                print("🔗 Edges table:", graph.edges.table().interactive())
            return

        if show_nodes:
            display(Markdown("### 📊 Nodes Table"))
            display_table(graph.nodes.table(), height=height)

        if show_edges:
            display(Markdown("### 🔗 Edges Table"))
            display_table(graph.edges.table(), height=height)

    except ImportError:
        print("⚠️  IPython not available - falling back to browser mode")
        if show_nodes:
            print("📊 Nodes table:", graph.nodes.table().interactive())
        if show_edges:
            print("🔗 Edges table:", graph.edges.table().interactive())
    except Exception as e:
        print(f"❌ Error displaying graph tables: {e}")


def embed_table_html(table, height=600, width="100%"):
    """
    Generate the HTML iframe code without automatically displaying it.

    This gives users full control over how and when to display the embedded table.

    Args:
        table: A groggy table object
        height: Height of the iframe (default: 600)
        width: Width of the iframe (default: "100%")

    Returns:
        str: HTML iframe code that can be used with display(HTML(...))

    Example:
        ```python
        from IPython.display import HTML, display
        from groggy.jupyter_utils import embed_table_html

        # Get the HTML without displaying it
        iframe_html = embed_table_html(table, height=800)

        # Display it later or combine with other HTML
        display(HTML(iframe_html))
        ```
    """
    iframe_html = table.interactive_embed()

    # Apply custom dimensions if specified
    if height != 600:
        iframe_html = iframe_html.replace('height="600px"', f'height="{height}px"')
    if width != "100%":
        iframe_html = iframe_html.replace('width="100%"', f'width="{width}"')

    return iframe_html
