import groggy as gr


def test_builder_sampler_unified():
    g = gr.generators.karate_club()

    b = gr.builder("random_10_1hop")
    nodes = b.sample_nodes(count=10, seed=42)
    nbh = b.neighbors(nodes, hops=1)
    b.emit_subgraphs(nbh, mode="unified")

    sampler = b.build_sampler()
    samples = g.view().sample(sampler)

    assert len(samples) == 1
    assert samples[0].node_count() > 0


def test_builder_sampler_map_per_item():
    g = gr.generators.karate_club()

    b = gr.builder("per_item_sample")
    nbh = b.neighbors(b.iterate_nodes(), hops=1)
    sampled = nbh.map(lambda b: b.sample_nodes(count=2, seed=1))
    b.emit_subgraphs(sampled, mode="per_item")

    sampler = b.build_sampler()
    samples = g.view().sample(sampler)

    assert len(samples) == g.node_count()
