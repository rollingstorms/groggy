from groggy.builder import AlgorithmBuilder
import pprint

def build_lpa_algorithm(max_iter=3):
    """Build LPA using the builder DSL with proper primitives."""
    builder = AlgorithmBuilder("custom_lpa")
    
    # Initialize each node with unique label (0, 1, 2, ...)
    labels = builder.init_nodes(unique=True)
    
    with builder.iterate(max_iter):
        labels = builder.core.neighbor_mode_update(
            labels,
            include_self=True,
            tie_break="lowest",
            ordered=True,
        )
    
    builder.attach_as("community", labels)
    return builder.build()

algo = build_lpa_algorithm(max_iter=3)
spec = algo._steps
for i, step in enumerate(spec):
    print(f"\nStep {i}: {step.get('type')}")
    pprint.pprint(step)
