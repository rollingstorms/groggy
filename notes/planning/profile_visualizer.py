#!/usr/bin/env python3
"""
Profile Visualization Helper

Creates ASCII bar charts from algorithm profile data to visualize
where time is spent in different phases of execution.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from collections import defaultdict


@dataclass
class ProfileEntry:
    """A single profile timing entry."""
    name: str
    time_ms: float
    parent: Optional[str] = None
    
    
class ProfileVisualizer:
    """Creates ASCII visualizations of profile data."""
    
    def __init__(self, width: int = 70):
        self.width = width
        self.bar_char = '█'
        self.empty_char = '░'
        
    def _format_time(self, ms: float) -> str:
        """Format time in appropriate units."""
        if ms < 0.001:
            return f"{ms * 1_000_000:.1f}ns"
        elif ms < 1.0:
            return f"{ms * 1000:.1f}μs"
        elif ms < 1000:
            return f"{ms:.2f}ms"
        else:
            return f"{ms / 1000:.2f}s"
    
    def _create_bar(self, ratio: float, width: int) -> str:
        """Create a horizontal bar showing the ratio."""
        filled = int(ratio * width)
        empty = width - filled
        return self.bar_char * filled + self.empty_char * empty
    
    def create_comparison_chart(
        self,
        entries: List[Tuple[str, float]],
        title: str = "Time Comparison"
    ) -> str:
        """Create a comparison chart of multiple entries.
        
        Args:
            entries: List of (name, time_ms) tuples
            title: Chart title
            
        Returns:
            ASCII art comparison chart
        """
        if not entries:
            return "No data to visualize"
        
        max_time = max(time for _, time in entries)
        max_name_len = max(len(name) for name, _ in entries)
        
        lines = []
        lines.append("=" * self.width)
        lines.append(title)
        lines.append("=" * self.width)
        lines.append("")
        
        for name, time_ms in entries:
            ratio = time_ms / max_time if max_time > 0 else 0
            bar_width = self.width - max_name_len - 20
            bar = self._create_bar(ratio, bar_width)
            time_str = self._format_time(time_ms)
            pct = (ratio * 100)
            
            lines.append(f"{name:<{max_name_len}} {bar} {time_str:>10} ({pct:5.1f}%)")
        
        lines.append("")
        lines.append(f"Total: {self._format_time(sum(t for _, t in entries))}")
        lines.append("=" * self.width)
        
        return "\n".join(lines)
    
    def create_breakdown_chart(
        self,
        total_time: float,
        breakdown: Dict[str, float],
        title: str = "Time Breakdown"
    ) -> str:
        """Create a breakdown chart showing components of total time.
        
        Args:
            total_time: Total execution time
            breakdown: Dict of {component: time_ms}
            title: Chart title
            
        Returns:
            ASCII art breakdown chart
        """
        lines = []
        lines.append("=" * self.width)
        lines.append(title)
        lines.append("=" * self.width)
        lines.append("")
        
        # Sort by time descending
        sorted_items = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        
        max_name_len = max(len(name) for name, _ in sorted_items) if sorted_items else 10
        bar_width = self.width - max_name_len - 25
        
        for name, time_ms in sorted_items:
            ratio = time_ms / total_time if total_time > 0 else 0
            bar = self._create_bar(ratio, bar_width)
            time_str = self._format_time(time_ms)
            pct = (ratio * 100)
            
            lines.append(f"  {name:<{max_name_len}} {bar} {time_str:>10} ({pct:5.1f}%)")
        
        lines.append("")
        lines.append(f"Total Time: {self._format_time(total_time)}")
        
        # Calculate overhead
        accounted = sum(breakdown.values())
        tolerance = max(total_time, accounted, 1.0) * 1e-6
        if accounted > total_time + tolerance:
            lines.append("Note: Component timers overlap (sum exceeds total).")
        else:
            overhead = total_time - accounted
            if abs(overhead) <= tolerance:
                overhead = 0.0

            if abs(overhead) > 0.001:
                overhead_pct = (overhead / total_time * 100) if total_time > 0 else 0
                lines.append(f"Unaccounted: {self._format_time(overhead)} ({overhead_pct:.1f}%)")
        
        lines.append("=" * self.width)
        
        return "\n".join(lines)
    
    def create_hierarchical_chart(
        self,
        profile_data: Dict[str, float],
        title: str = "Profile Hierarchy"
    ) -> str:
        """Create a hierarchical chart from nested profile data.
        
        Args:
            profile_data: Dict with keys like "parent.child" showing hierarchy
            title: Chart title
            
        Returns:
            ASCII art hierarchical chart
        """
        lines = []
        lines.append("=" * self.width)
        lines.append(title)
        lines.append("=" * self.width)
        lines.append("")
        
        # Build hierarchy
        hierarchy = {}
        for key, time_ms in profile_data.items():
            parts = key.split('.')
            current = hierarchy
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {'_children': {}, '_time': 0}
                current = current[part]['_children']
            
            final = parts[-1]
            if final not in current:
                current[final] = {'_children': {}, '_time': time_ms}
            else:
                current[final]['_time'] += time_ms
        
        def print_level(node_dict, indent=0, prefix=""):
            """Recursively print hierarchy."""
            items = sorted(node_dict.items(), key=lambda x: x[1]['_time'], reverse=True)
            
            for i, (name, data) in enumerate(items):
                is_last = i == len(items) - 1
                time_ms = data['_time']
                time_str = self._format_time(time_ms)
                
                connector = "└─ " if is_last else "├─ "
                line = f"{prefix}{connector}{name}: {time_str}"
                lines.append(line)
                
                # Recurse for children
                if data['_children']:
                    child_prefix = prefix + ("   " if is_last else "│  ")
                    print_level(data['_children'], indent + 1, child_prefix)
        
        print_level(hierarchy)
        
        lines.append("")
        lines.append("=" * self.width)
        
        return "\n".join(lines)
    
    def create_stacked_chart(
        self,
        stages: List[Tuple[str, float]],
        total_time: float,
        title: str = "Execution Stages"
    ) -> str:
        """Create a stacked horizontal bar showing execution stages.
        
        Args:
            stages: List of (stage_name, time_ms) in execution order
            total_time: Total execution time
            title: Chart title
            
        Returns:
            ASCII art stacked bar chart
        """
        lines = []
        lines.append("=" * self.width)
        lines.append(title)
        lines.append("=" * self.width)
        lines.append("")
        
        # Create the stacked bar
        bar_width = self.width - 10
        segments = []
        
        for name, time_ms in stages:
            ratio = time_ms / total_time if total_time > 0 else 0
            segment_width = int(ratio * bar_width)
            segments.append((name, segment_width, time_ms, ratio))
        
        # Draw the bar
        bar_parts = []
        for name, width, time_ms, ratio in segments:
            if width > 0:
                bar_parts.append(self.bar_char * width)
            else:
                bar_parts.append(self.empty_char)
        
        bar = "".join(bar_parts)
        # Pad or truncate to exact width
        if len(bar) < bar_width:
            bar += self.empty_char * (bar_width - len(bar))
        bar = bar[:bar_width]
        
        lines.append(f"  {bar}")
        lines.append("")
        
        # Legend
        lines.append("  Stages:")
        for name, width, time_ms, ratio in segments:
            pct = ratio * 100
            time_str = self._format_time(time_ms)
            lines.append(f"    • {name}: {time_str} ({pct:.1f}%)")
        
        lines.append("")
        lines.append(f"  Total: {self._format_time(total_time)}")
        lines.append("=" * self.width)
        
        return "\n".join(lines)


def visualize_groggy_profile(stats: Dict, algorithm_name: str = "Algorithm") -> str:
    """Create visualizations from Groggy profile stats.
    
    Args:
        stats: Profile stats dict from sg.apply(..., return_profile=True)
        algorithm_name: Name of the algorithm
        
    Returns:
        Formatted visualization string
    """
    viz = ProfileVisualizer(width=75)
    
    output = []
    
    # Overall timing
    output.append(viz.create_comparison_chart(
        [
            ("Build Time", stats.get('build_time', 0) * 1000),
            ("Run Time", stats.get('run_time', 0) * 1000),
            ("Clone Time", stats.get('subgraph_clone_time', 0) * 1000),
        ],
        title=f"{algorithm_name} - Overall Timing"
    ))
    output.append("")
    
    # Detailed timers if available
    timers = stats.get('timers', {})
    if timers:
        raw_breakdown = {k: v * 1000 for k, v in timers.items()}
        aggregated_breakdown: Dict[str, float] = {}
        for key, value in raw_breakdown.items():
            root = key.split('.', 1)[0]
            if root in aggregated_breakdown:
                aggregated_breakdown[root] = max(aggregated_breakdown[root], value)
            else:
                aggregated_breakdown[root] = value
        timer_breakdown = aggregated_breakdown
        total_run_ms = stats.get('run_time', 0) * 1000
        max_component_ms = max(timer_breakdown.values(), default=0.0)
        if max_component_ms > total_run_ms:
            total_run_ms = max_component_ms
        accounted_ms = sum(timer_breakdown.values())
        if accounted_ms < total_run_ms:
            accounted_ms = total_run_ms
        output.append(viz.create_breakdown_chart(
            total_run_ms,
            timer_breakdown,
            title=f"{algorithm_name} - Detailed Breakdown"
        ))
        output.append("")
        
        # Hierarchical view
        output.append(viz.create_hierarchical_chart(
            timer_breakdown,
            title=f"{algorithm_name} - Call Hierarchy"
        ))
        output.append("")
    
    # Execution stages
    stages = []
    if 'build_time' in stats:
        stages.append(("Build Pipeline", stats['build_time'] * 1000))
    if 'subgraph_clone_time' in stats:
        stages.append(("Clone Subgraph", stats['subgraph_clone_time'] * 1000))
    if 'run_time' in stats:
        stages.append(("Run Algorithm", stats['run_time'] * 1000))
    
    if stages:
        total = sum(t for _, t in stages)
        output.append(viz.create_stacked_chart(
            stages,
            total,
            title=f"{algorithm_name} - Execution Pipeline"
        ))
    
    return "\n".join(output)


# Example usage and tests
if __name__ == '__main__':
    viz = ProfileVisualizer()
    
    # Example 1: Simple comparison
    print(viz.create_comparison_chart(
        [
            ("Serial PageRank", 3.09),
            ("Parallel PageRank", 21.15),
            ("FFI Overhead", 2.0),
        ],
        title="PageRank Performance Comparison"
    ))
    print()
    
    # Example 2: Breakdown
    print(viz.create_breakdown_chart(
        21.15,
        {
            "HashMap Operations": 15.0,
            "HashMap Merging": 4.0,
            "Convergence Check": 1.5,
            "Buffer Management": 0.5,
        },
        title="Parallel PageRank Breakdown"
    ))
    print()
    
    # Example 3: Hierarchical
    print(viz.create_hierarchical_chart(
        {
            "community.lpa": 5.0,
            "community.lpa.init": 0.5,
            "community.lpa.iterations": 4.0,
            "community.lpa.iterations.count_labels": 3.0,
            "community.lpa.iterations.update": 1.0,
            "community.lpa.write_attrs": 0.5,
        },
        title="LPA Profile Hierarchy"
    ))
    print()
    
    # Example 4: Stacked stages
    print(viz.create_stacked_chart(
        [
            ("Build", 0.5),
            ("Clone", 0.3),
            ("Execute", 4.0),
            ("Write Results", 0.2),
        ],
        5.0,
        title="Algorithm Execution Stages"
    ))
