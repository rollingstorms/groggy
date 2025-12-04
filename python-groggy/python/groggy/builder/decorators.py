"""
Decorator system for defining algorithms with the builder DSL.

This module provides decorators that make algorithm definitions cleaner
and more declarative, similar to JAX or PyTorch Lightning.
"""

from functools import wraps
from typing import Any, Callable, Optional, Union

from groggy.builder.algorithm_builder import AlgorithmBuilder
from groggy.builder.varhandle import VarHandle


def algorithm(name_or_func: Optional[Union[str, Callable]] = None):
    """
    Decorator for defining algorithms with the builder DSL.

    The decorated function receives a GraphHandle (sG) as first argument and should
    return either:
    - A VarHandle to use as output (will be attached with algorithm name)
    - None (if output is attached manually via builder.attr.save)

    Args:
        name_or_func: Optional algorithm name (defaults to function name)
                      Or the function itself if used without parentheses

    Note:
        By convention, use `sG` as the parameter name to indicate it's a subgraph view.

    Usage:
        >>> # With explicit name
        >>> @algorithm("my_pagerank")
        ... def pagerank(sG, damping=0.85, max_iter=100):
        ...     ranks = sG.nodes(1.0 / sG.N)
        ...     deg = ranks.degrees()
        ...
        ...     with sG.builder.iter.loop(max_iter):
        ...         neighbor_sum = sG @ (ranks / (deg + 1e-9))
        ...         ranks = sG.builder.var("ranks", damping * neighbor_sum + (1 - damping) / sG.N)
        ...
        ...     return ranks.normalize()
        ...
        >>> pr_algo = pagerank(damping=0.9)
        >>> result = subgraph.apply(pr_algo)

        >>> # Without parentheses (uses function name)
        >>> @algorithm
        ... def label_propagation(sG, max_iter=10):
        ...     labels = sG.nodes(unique=True)
        ...     with sG.builder.iter.loop(max_iter):
        ...         labels = sG.builder.graph_ops.neighbor_mode_update(labels)
        ...     return labels
    """

    # Handle both @algorithm and @algorithm("name") syntax
    if callable(name_or_func):
        # Used as @algorithm without parentheses
        return _create_algorithm_wrapper(name_or_func.__name__, name_or_func)
    else:
        # Used as @algorithm("name") with explicit name
        algo_name = name_or_func

        def decorator(func: Callable):
            # Use provided name or fall back to function name
            final_name = algo_name if algo_name else func.__name__
            return _create_algorithm_wrapper(final_name, func)

        return decorator


def _create_algorithm_wrapper(algo_name: str, func: Callable):
    """
    Internal helper to create the algorithm wrapper.

    Args:
        algo_name: Name for the algorithm
        func: User function to wrap

    Returns:
        Wrapped function that builds and returns an algorithm
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create builder
        builder = AlgorithmBuilder(algo_name)

        # Create graph handle and inject builder reference
        sG = builder.graph()

        # Call user function with graph handle + remaining args
        result = func(sG, *args, **kwargs)

        # If result is a VarHandle, attach it as output
        if isinstance(result, VarHandle):
            builder.attr.save(algo_name, result)
        elif result is not None:
            raise ValueError(
                f"Algorithm function must return VarHandle or None, got {type(result)}"
            )

        # Build and return algorithm
        return builder.build()

    # Preserve function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__

    return wrapper


def compiled(fn: Callable):
    """
    Decorator for JIT-compiled algorithms (future feature).

    This is a placeholder for future JIT compilation support (Phase 6).
    Currently, it just passes through the function unchanged.

    Usage:
        >>> @compiled
        ... @algorithm
        ... def fast_pagerank(G, damping=0.85):
        ...     # Algorithm definition
        ...     ...

    Example:
        >>> @compiled
        ... @algorithm("optimized_pr")
        ... def pagerank(sG, damping=0.85, max_iter=100):
        ...     ranks = sG.nodes(1.0 / sG.N)
        ...     # ... rest of algorithm
        ...     return ranks
    """
    # For now, just pass through
    # Future: Trigger JIT compilation when algorithm is built
    return fn


def traced(fn: Callable):
    """
    Decorator to enable tracing/debugging for algorithm execution (future feature).

    This is a placeholder for future debugging infrastructure.

    Usage:
        >>> @traced
        ... @algorithm
        ... def debug_pagerank(sG):
        ...     # Algorithm will print execution trace
        ...     ...
    """
    # Placeholder for future tracing infrastructure
    return fn


# Backward compatibility alias
def builder_algorithm(name: Optional[str] = None):
    """
    Backward compatibility alias for @algorithm decorator.

    Deprecated: Use @algorithm instead.
    """
    import warnings

    warnings.warn(
        "@builder_algorithm is deprecated, use @algorithm instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return algorithm(name)
