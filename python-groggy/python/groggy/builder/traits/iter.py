"""
Control flow and iteration constructs.

This module provides IterOps for loops, convergence, and update strategies.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groggy.builder.algorithm_builder import AlgorithmBuilder
    from groggy.builder.varhandle import VarHandle


class IterOps:
    """Control flow and iteration constructs."""

    def __init__(self, builder: "AlgorithmBuilder"):
        """
        Initialize iteration operations.

        Args:
            builder: Parent algorithm builder
        """
        self.builder = builder

    def loop(self, count: int):
        """
        Fixed iteration loop.

        Args:
            count: Number of iterations

        Returns:
            Context manager for loop body

        Example:
            >>> with builder.iter.loop(100):
            ...     neighbor_sum = G @ ranks
            ...     ranks = builder.var("ranks", 0.85 * neighbor_sum + 0.15 / G.N)
        """
        # Delegate to builder.iterate for now (backward compatibility)
        return self.builder.iterate(count)

    def until_converged(
        self, watched: "VarHandle", tol: float = 1e-6, max_iter: int = 1000
    ):
        """
        Loop until convergence (future feature).

        Args:
            watched: Variable to watch for convergence
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Example:
            >>> with builder.iter.until_converged(ranks, tol=1e-6):
            ...     ranks = update_ranks(ranks)
        """
        # Placeholder for future IR-based convergence detection
        # For now, fall back to fixed iteration
        import warnings

        warnings.warn(
            "until_converged() not yet implemented, falling back to fixed iteration",
            FutureWarning,
        )
        return self.builder.iterate(max_iter)

    def strategy(self, mode: str = "sync"):
        """
        Set update strategy (future feature).

        Args:
            mode: 'sync' (batch updates) or 'async' (immediate updates)

        Note: Currently only 'sync' is implemented for most operations.
              Use graph.neighbor_mode_update() for async LPA-style updates.
        """
        # This is metadata that could affect how future operations execute
        # For now, it's a no-op placeholder
        if mode not in ["sync", "async"]:
            raise ValueError(f"Invalid strategy mode: {mode}")
        # Future: store this in builder state to affect subsequent operations
        pass
