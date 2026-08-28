"""Regression contract for direct GraphML-to-CSR loading."""

from drone_emission.roadnet.sparse_graph import SparseGraph


def test_graphml_streaming_preserves_sorted_nodes_and_min_parallel_edge(tmp_path):
    graphml = tmp_path / "tiny.graphml"
    graphml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="x" for="node" attr.name="x" attr.type="double"/>
  <key id="y" for="node" attr.name="y" attr.type="double"/>
  <key id="len" for="edge" attr.name="length" attr.type="double"/>
  <graph edgedefault="directed">
    <node id="20"><data key="x">114.2</data><data key="y">22.2</data></node>
    <node id="10"><data key="x">114.1</data><data key="y">22.1</data></node>
    <edge source="20" target="10"><data key="len">5</data></edge>
    <edge source="20" target="10"><data key="len">3</data></edge>
    <edge source="10" target="20"><data key="len">7</data></edge>
  </graph>
</graphml>
""",
        encoding="utf-8",
    )

    graph = SparseGraph.from_graphml(str(graphml))

    assert graph.num_nodes == 2
    assert graph.num_edges == 2
    assert graph.get_node_osmid(0) == 10
    assert graph.get_node_coords(0) == (22.1, 114.1)
    assert graph._dist_matrix[1, 0] == 3.0
    assert graph._dist_matrix[0, 1] == 7.0
    assert graph.total_length == 10.0
