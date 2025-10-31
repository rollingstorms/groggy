"""
High-level Pipeline API for algorithm composition.
"""

from typing import List, Union, Dict
import time
from groggy import _groggy
from groggy.algorithms.base import AlgorithmHandle


def apply(subgraph, algorithm_or_pipeline, persist=True, return_profile=False):
    """
    Apply an algorithm or pipeline to a subgraph.

    This convenience helper accepts the same inputs as ``Subgraph.apply``:

    * a single ``AlgorithmHandle`` (runs one algorithm)
    * a list/tuple of handles (runs them sequentially)
    * an existing ``Pipeline`` instance

    Args:
        subgraph: The subgraph to process
        algorithm_or_pipeline: Algorithm handle, list of handles, or ``Pipeline``
        persist: Whether to persist algorithm results as attributes (default: True)
        return_profile: If True, return (subgraph, profile_dict); otherwise just subgraph (default: False)

    Returns:
        Processed subgraph with algorithm results (or tuple with profile if return_profile=True)

    Example:
        >>> from groggy import algorithms, builder, apply
        >>> result = apply(subgraph, algorithms.centrality.pagerank())
        >>>
        >>> result = apply(subgraph, [
        ...     algorithms.centrality.pagerank(max_iter=20, output_attr="pr"),
        ...     algorithms.pathfinding.bfs(start_attr="is_start", output_attr="dist")
        ... ])
        >>>
        >>> # Control persistence and get profiling info
        >>> result, profile = apply(subgraph, algo, persist=False, return_profile=True)
        >>> print(f"Algorithm took {profile['run_time']:.6f}s")
        >>>
        >>> b = builder("degree")
        >>> nodes = b.init_nodes(default=0.0)
        >>> b.attach_as("degree", b.node_degrees(nodes))
        >>> custom = b.build()
        >>> result = apply(subgraph, custom)
    """
    if isinstance(algorithm_or_pipeline, Pipeline):
        return algorithm_or_pipeline(subgraph, persist=persist, return_profile=return_profile)
    elif isinstance(algorithm_or_pipeline, list):
        pipe = Pipeline(algorithm_or_pipeline)
        return pipe(subgraph, persist=persist, return_profile=return_profile)
    elif isinstance(algorithm_or_pipeline, AlgorithmHandle):
        pipe = Pipeline([algorithm_or_pipeline])
        return pipe(subgraph, persist=persist, return_profile=return_profile)
    else:
        raise TypeError(
            f"Expected Pipeline, AlgorithmHandle, or list, got {type(algorithm_or_pipeline)}"
        )


class Pipeline:
    """
    High-level pipeline for composing and executing algorithms.
    
    A pipeline represents a sequence of algorithms that will be executed
    in order, with the output of each algorithm becoming the input to the next.
    
    Example:
        >>> from groggy import pipeline
        >>> from groggy.algorithms import centrality, pathfinding
        >>>
        >>> pipe = pipeline([
        ...     centrality.pagerank(max_iter=20, output_attr="pr"),
        ...     pathfinding.bfs(start_attr="is_start", output_attr="dist")
        ... ])
        >>> result = subgraph.apply(pipe)     # fluent style
        >>> # Or equivalently:
        >>> result = pipe(subgraph)           # callable pipeline
    """
    
    def __init__(self, algorithms: List[Union[AlgorithmHandle, dict]]):
        """
        Create a pipeline from a list of algorithms.
        
        Args:
            algorithms: List of AlgorithmHandle objects or spec dicts
        """
        self.algorithms = algorithms
        self._handle = None
        self._last_profile = None
        self._build_timings: Dict[str, float] = {}

        start = time.perf_counter()
        self._validate_algorithms()
        self._build_timings["pipeline.validate"] = time.perf_counter() - start
    
    def _validate_algorithms(self):
        """Validate all algorithms in the pipeline."""
        for i, algo in enumerate(self.algorithms):
            if isinstance(algo, AlgorithmHandle):
                # Validate if it's a RustAlgorithmHandle
                if hasattr(algo, 'validate'):
                    try:
                        algo.validate()
                    except ValueError as e:
                        raise ValueError(f"Algorithm at index {i} is invalid: {e}")
            elif isinstance(algo, dict):
                # Already a spec dict, assume it's valid
                pass
            else:
                raise TypeError(
                    f"Algorithm at index {i} must be an AlgorithmHandle or dict, "
                    f"got {type(algo)}"
                )
    
    def _build_spec(self) -> list:
        """Build the FFI pipeline spec."""
        spec = []
        for algo in self.algorithms:
            if isinstance(algo, AlgorithmHandle):
                spec.append(algo.to_spec())
            else:
                # Assume it's already a dict spec
                spec.append(algo)
        return spec
    
    def _ensure_built(self):
        """Ensure the pipeline is built in the FFI layer."""
        if self._handle is None:
            timings: Dict[str, float] = dict(self._build_timings)
            build_total_start = time.perf_counter()

            spec_start = time.perf_counter()
            spec = self._build_spec()
            timings["pipeline.build_spec"] = time.perf_counter() - spec_start

            build_handle_start = time.perf_counter()
            handle = _groggy.pipeline.build_pipeline(spec)
            timings["pipeline.build_handle"] = time.perf_counter() - build_handle_start

            timings["pipeline.build_total"] = time.perf_counter() - build_total_start

            self._handle = handle
            self._build_timings = timings
    
    def run(self, subgraph, persist=True, return_profile=False):
        """
        Run the pipeline on a subgraph.
        
        Args:
            subgraph: The subgraph to process
            persist: Whether to persist algorithm results as attributes (default: True)
            return_profile: If True, return (subgraph, profile_dict); otherwise just subgraph (default: False)
            
        Returns:
            Processed subgraph with algorithm results (or tuple with profile if return_profile=True)
        """
        self._ensure_built()
        run_call_start = time.perf_counter()
        result_subgraph, profile_dict = _groggy.pipeline.run_pipeline(
            self._handle, subgraph, persist_results=persist
        )
        run_call_elapsed = time.perf_counter() - run_call_start

        python_timings: Dict[str, float] = dict(self._build_timings)
        python_timings["pipeline.run_call"] = run_call_elapsed

        ffi_total = 0.0
        if isinstance(profile_dict, dict):
            ffi_section = profile_dict.get("ffi_timers")
            if isinstance(ffi_section, dict):
                ffi_total = float(ffi_section.get("ffi.total", 0.0))
                # Ensure ffi section values are floats
                for key, value in list(ffi_section.items()):
                    try:
                        ffi_section[key] = float(value)
                    except (TypeError, ValueError):
                        ffi_section[key] = 0.0

        python_timings["pipeline.run_python_overhead"] = max(0.0, run_call_elapsed - ffi_total)

        if isinstance(profile_dict, dict):
            profile_dict["python_pipeline_timings"] = python_timings
            profile_dict["python_apply"] = {
                "collect_spec": self._build_timings.get("pipeline.build_spec", 0.0),
                "build_pipeline_py": self._build_timings.get("pipeline.build_total", 0.0),
                "run_pipeline_py": run_call_elapsed,
                "run_pipeline_overhead": python_timings.get("pipeline.run_python_overhead", 0.0),
                "drop_pipeline_py": 0.0,
                "create_subgraph_obj_py": 0.0,
                "apply_remaining_py": 0.0,
            }
        self._build_timings = python_timings
        
        # Store the profile for last_profile() method
        self._last_profile = profile_dict
        
        if return_profile:
            return result_subgraph, profile_dict
        else:
            return result_subgraph
    
    def last_profile(self):
        """
        Get the profile information from the last pipeline run.
        
        Returns:
            Dictionary with profiling information from the last run, or None if no run yet
        """
        return self._last_profile
    
    def __call__(self, subgraph, persist=True, return_profile=False):
        """
        Allow pipeline to be called as a function.
        
        Args:
            subgraph: The subgraph to process
            persist: Whether to persist algorithm results as attributes (default: True)
            return_profile: If True, return (subgraph, profile_dict); otherwise just subgraph (default: False)
            
        Returns:
            Processed subgraph (or tuple with profile if return_profile=True)
        """
        return self.run(subgraph, persist=persist, return_profile=return_profile)
    
    def __del__(self):
        """Clean up the FFI pipeline handle."""
        if self._handle is not None:
            try:
                _groggy.pipeline.drop_pipeline(self._handle)
            except:
                pass  # Ignore errors during cleanup
    
    def __repr__(self) -> str:
        """String representation."""
        algo_names = []
        for algo in self.algorithms:
            if isinstance(algo, AlgorithmHandle):
                algo_names.append(algo.id)
            elif isinstance(algo, dict):
                algo_names.append(algo.get("id", "unknown"))
            else:
                algo_names.append("unknown")
        
        return f"Pipeline([{', '.join(algo_names)}])"
    
    def __len__(self) -> int:
        """Get the number of algorithms in the pipeline."""
        return len(self.algorithms)


def pipeline(algorithms: List[Union[AlgorithmHandle, dict]]) -> Pipeline:
    """
    Create a pipeline from a list of algorithms.
    
    Args:
        algorithms: List of algorithm handles or spec dicts
        
    Returns:
        Pipeline object ready to execute
        
    Example:
        >>> from groggy import pipeline, algorithms
        >>>
        >>> pipe = pipeline([
        ...     algorithms.centrality.pagerank(max_iter=20, output_attr="pr"),
        ...     algorithms.community.lpa(output_attr="community")
        ... ])
        >>> result = pipe(subgraph)
    """
    return Pipeline(algorithms)
