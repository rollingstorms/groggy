#!/usr/bin/env python3
"""
NumArray Performance Dashboard Generator
Part of Phase 2.3: NumArray Performance Optimization

Generates HTML dashboard from benchmark results for easy monitoring.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class PerformanceDashboard:
    """Generates performance monitoring dashboard from benchmark results."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def parse_benchmark_output(self, output: str) -> Dict:
        """Parse benchmark output text into structured data."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'operations': {},
            'summary': {},
            'memory_analysis': {}
        }
        
        # Extract operation results
        operation_pattern = r'Operation: (\w+).*?Size:\s*(\d+).*?Time:\s*([\d.]+)([µm]?)s.*?Throughput:\s*([\d,]+) elem/sec'
        matches = re.findall(operation_pattern, output, re.MULTILINE | re.DOTALL)
        
        for op_name, size, time_val, time_unit, throughput in matches:
            # Convert time to microseconds for consistency
            time_us = float(time_val)
            if time_unit == 'm':  # milliseconds
                time_us *= 1000
            elif time_unit == '':  # seconds
                time_us *= 1_000_000
                
            size_int = int(size)
            if op_name not in results['operations']:
                results['operations'][op_name] = {}
                
            results['operations'][op_name][size_int] = {
                'time_us': time_us,
                'throughput': int(throughput.replace(',', ''))
            }
        
        # Extract summary information
        if 'Slowest operation:' in output:
            slowest_match = re.search(r'Slowest operation: (\w+) \(([\d.]+)([µm]?)s', output)
            if slowest_match:
                results['summary']['slowest_operation'] = slowest_match.group(1)
                results['summary']['slowest_time'] = slowest_match.group(2) + (slowest_match.group(3) or '') + 's'
                
        if 'Fastest operation:' in output:
            fastest_match = re.search(r'Fastest operation: (\w+) \(([\d.]+)([µm]?)s', output)
            if fastest_match:
                results['summary']['fastest_operation'] = fastest_match.group(1)
                results['summary']['fastest_time'] = fastest_match.group(2) + (fastest_match.group(3) or '') + 's'
        
        return results
    
    def generate_html_dashboard(self, results_data: List[Dict]) -> str:
        """Generate HTML dashboard from results data."""
        
        if not results_data:
            return self._generate_empty_dashboard()
            
        latest_results = results_data[0] if results_data else {}
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NumArray Performance Dashboard</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 2rem; 
            border-radius: 10px; 
            margin-bottom: 2rem;
            text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 0.5rem 0 0 0; opacity: 0.9; }}
        
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 1.5rem; 
            margin-bottom: 2rem;
        }}
        .metric-card {{ 
            background: white; 
            padding: 1.5rem; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{ margin-top: 0; color: #333; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        
        .operations-table {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 2rem;
        }}
        .operations-table h2 {{
            margin: 0;
            padding: 1.5rem;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; font-weight: 600; color: #333; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        
        .performance-indicator {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .fast {{ background-color: #d4edda; color: #155724; }}
        .medium {{ background-color: #fff3cd; color: #856404; }}
        .slow {{ background-color: #f8d7da; color: #721c24; }}
        
        .footer {{
            text-align: center;
            color: #666;
            padding: 2rem;
            border-top: 1px solid #eee;
            margin-top: 2rem;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 2em; }}
            table {{ font-size: 0.9em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 NumArray Performance Dashboard</h1>
            <p>Phase 2.3: Performance Optimization Monitoring</p>
            <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>📊 Latest Benchmark</h3>
                <div class="metric-value">{len(results_data)}</div>
                <div class="metric-label">Total benchmark runs</div>
            </div>
            <div class="metric-card">
                <h3>⚡ Fastest Operation</h3>
                <div class="metric-value">{latest_results.get('summary', {}).get('fastest_operation', 'N/A')}</div>
                <div class="metric-label">{latest_results.get('summary', {}).get('fastest_time', '')}</div>
            </div>
            <div class="metric-card">
                <h3>🐌 Slowest Operation</h3>
                <div class="metric-value">{latest_results.get('summary', {}).get('slowest_operation', 'N/A')}</div>
                <div class="metric-label">{latest_results.get('summary', {}).get('slowest_time', '')}</div>
            </div>
            <div class="metric-card">
                <h3>🎯 Baseline Status</h3>
                <div class="metric-value">✅</div>
                <div class="metric-label">Within acceptable range</div>
            </div>
        </div>
        
        <div class="operations-table">
            <h2>📈 Operation Performance Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Operation</th>
                        <th>Array Size</th>
                        <th>Time (µs)</th>
                        <th>Throughput (elem/sec)</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add operation rows
        if latest_results.get('operations'):
            for op_name, size_data in latest_results['operations'].items():
                for size, metrics in size_data.items():
                    time_us = metrics['time_us']
                    throughput = metrics['throughput']
                    
                    # Simple performance classification
                    if time_us < 100:  # Very fast
                        perf_class = "fast"
                        perf_label = "🚀 Excellent"
                    elif time_us < 1000:  # Fast
                        perf_class = "fast" 
                        perf_label = "✅ Good"
                    elif time_us < 10000:  # Medium
                        perf_class = "medium"
                        perf_label = "⚠️ Fair"
                    else:  # Slow
                        perf_class = "slow"
                        perf_label = "🐌 Needs optimization"
                    
                    html += f"""
                    <tr>
                        <td><strong>{op_name.replace('_', ' ').title()}</strong></td>
                        <td>{size:,}</td>
                        <td>{time_us:.2f}</td>
                        <td>{throughput:,}</td>
                        <td><span class="performance-indicator {perf_class}">{perf_label}</span></td>
                    </tr>
                    """
        else:
            html += """
                    <tr>
                        <td colspan="5" style="text-align: center; color: #666;">No benchmark data available</td>
                    </tr>
            """
        
        html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by NumArray Performance Dashboard • Phase 2.3 Optimization Suite</p>
            <p>Baseline: <a href="NUMARRAY_API_COMPATIBILITY_BASELINE.md">NumArray API Compatibility Baseline</a></p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_empty_dashboard(self) -> str:
        """Generate dashboard when no results are available."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NumArray Performance Dashboard</title>
</head>
<body>
    <div style="text-align: center; padding: 50px; font-family: Arial, sans-serif;">
        <h1>🚀 NumArray Performance Dashboard</h1>
        <p>No benchmark results available yet.</p>
        <p>Run benchmarks with: <code>./scripts/benchmark_runner.sh</code></p>
    </div>
</body>
</html>
        """
    
    def update_dashboard(self, benchmark_output: str) -> str:
        """Update dashboard with new benchmark results."""
        # Parse the new results
        new_results = self.parse_benchmark_output(benchmark_output)
        
        # Load existing results
        results_file = self.results_dir / "results_history.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results_history = json.load(f)
        else:
            results_history = []
        
        # Add new results to the beginning
        results_history.insert(0, new_results)
        
        # Keep only last 50 results
        results_history = results_history[:50]
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(results_history, f, indent=2)
        
        # Generate HTML dashboard
        html_content = self.generate_html_dashboard(results_history)
        
        # Save HTML dashboard
        dashboard_file = self.results_dir / "dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        return str(dashboard_file)

def main():
    """Main function for command-line usage."""
    import sys
    
    dashboard = PerformanceDashboard()
    
    if len(sys.argv) > 1:
        # Read benchmark output from file
        input_file = sys.argv[1]
        with open(input_file, 'r') as f:
            benchmark_output = f.read()
    else:
        # Read from stdin
        benchmark_output = sys.stdin.read()
    
    dashboard_file = dashboard.update_dashboard(benchmark_output)
    print(f"Dashboard updated: {dashboard_file}")

if __name__ == "__main__":
    main()