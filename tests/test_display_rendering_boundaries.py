"""Public table HTML retains exact extreme integers and escaped captured text."""
from groggy import Graph


def test_table_html_keeps_full_signed_integer_and_escaped_frozen_text():
    graph = Graph()
    graph.add_node(integer=-(2**63), message="猫<&")
    table = graph.read_view().nodes_table()
    head = graph.current_state_id
    html = table._repr_html_()
    assert "-9,223,372,036,854,775,808" in html
    assert "猫&lt;&amp;" in html
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "integer") == -(2**63)
    graph.set_node_attr(0, "message", "changed")
    assert "猫&lt;&amp;" in table._repr_html_()
