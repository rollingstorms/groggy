#!/usr/bin/env python3
"""
Test Results Aggregator for Generated Groggy Test Scripts

This script runs all generated test scripts and creates a comprehensive
report comparing the results with our previous dynamic testing.
"""

import os
import subprocess
import sys
import pandas as pd
import json
from datetime import datetime
import re

def run_test_script(script_path):
    """Run a single test script and parse its output"""
    print(f"🧪 Running {os.path.basename(script_path)}...")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        output = result.stdout
        
        # Parse the results from the output
        object_name = os.path.basename(script_path).replace('test_', '').replace('.py', '')
        
        # Extract success rate using regex
        rate_match = re.search(r'(\d+)/(\d+) methods working \((\d+\.\d+)%\)', output)
        if rate_match:
            working = int(rate_match.group(1))
            total = int(rate_match.group(2))
            rate = float(rate_match.group(3))
        else:
            working = total = rate = 0
        
        # Extract working and failing methods
        working_methods = []
        failing_methods = []
        
        # Look for method results in output
        for line in output.split('\n'):
            if '✅' in line and '() →' in line:
                method_match = re.search(r'✅ (\w+)\(\)', line)
                if method_match:
                    working_methods.append(method_match.group(1))
            elif '❌' in line and '() →' in line:
                method_match = re.search(r'❌ (\w+)\(\)', line)
                if method_match:
                    failing_methods.append(method_match.group(1))
        
        return {
            'object_name': object_name,
            'success': result.returncode == 0,
            'working_methods': working,
            'total_methods': total,
            'success_rate': rate,
            'working_method_names': working_methods,
            'failing_method_names': failing_methods,
            'output': output,
            'error': result.stderr if result.returncode != 0 else None
        }
        
    except subprocess.TimeoutExpired:
        return {
            'object_name': object_name,
            'success': False,
            'error': 'Timeout after 5 minutes',
            'output': None
        }
    except Exception as e:
        return {
            'object_name': object_name, 
            'success': False,
            'error': str(e),
            'output': None
        }

def load_previous_csv_results():
    """Load our previous comprehensive API test results from CSV"""
    csv_file = 'groggy_comprehensive_api_test_results.csv'
    
    if not os.path.exists(csv_file):
        print(f"⚠️ Previous results CSV not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Group by object_type and calculate success rates
    summary = df.groupby('object_type').agg({
        'success': ['count', 'sum'],
        'method_name': 'count'
    }).round(1)
    
    # Flatten column names
    summary.columns = ['total_methods', 'working_methods', 'total_methods_check']
    summary['success_rate'] = (summary['working_methods'] / summary['total_methods'] * 100).round(1)
    
    # Get method lists
    previous_results = {}
    for obj_type in df['object_type'].unique():
        obj_data = df[df['object_type'] == obj_type]
        working = obj_data[obj_data['success'] == True]['method_name'].tolist()
        failing = obj_data[obj_data['success'] == False]['method_name'].tolist()
        
        previous_results[obj_type] = {
            'working_methods': len(working),
            'total_methods': len(obj_data),
            'success_rate': len(working) / len(obj_data) * 100,
            'working_method_names': working,
            'failing_method_names': failing
        }
    
    return previous_results

def compare_results(generated_results, previous_results):
    """Compare generated test results with previous dynamic results"""
    
    comparison = []
    
    for result in generated_results:
        obj_name = result['object_name']
        
        # Map object names to match previous results
        obj_mapping = {
            'grapharray': 'GraphArray',
            'graph': 'Graph', 
            'nodestable': 'NodesTable',
            'edgestable': 'EdgesTable',
            'graphmatrix': 'GraphMatrix',
            'graphtable': 'GraphTable',
            'subgraph': 'Subgraph',
            'neighborhoodresult': 'NeighborhoodResult',
            'basetable': 'BaseTable'
        }
        
        mapped_name = obj_mapping.get(obj_name.lower(), obj_name)
        previous = previous_results.get(mapped_name, {}) if previous_results else {}
        
        comparison_data = {
            'object_name': mapped_name,
            'generated_working': result.get('working_methods', 0),
            'generated_total': result.get('total_methods', 0), 
            'generated_rate': result.get('success_rate', 0),
            'previous_working': previous.get('working_methods', 0),
            'previous_total': previous.get('total_methods', 0),
            'previous_rate': previous.get('success_rate', 0),
            'improvement': result.get('success_rate', 0) - previous.get('success_rate', 0),
            'generated_success': result.get('success', False)
        }
        
        comparison.append(comparison_data)
    
    return comparison

def generate_report(generated_results, previous_results, comparison):
    """Generate a comprehensive comparison report"""
    
    report = f"""
# Groggy API Testing Results Comparison

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report compares the results of generated individual test scripts 
versus our previous dynamic comprehensive API testing.

## Executive Summary

"""
    
    if previous_results:
        total_generated = sum(r['generated_working'] for r in comparison)
        total_previous = sum(r['previous_working'] for r in comparison)
        
        report += f"""
- **Generated Scripts**: {len([r for r in generated_results if r.get('success', False)])} successful out of {len(generated_results)}
- **Working Methods**: {total_generated} (Generated) vs {total_previous} (Previous Dynamic)
- **Average Improvement**: {sum(r['improvement'] for r in comparison if r['improvement'] is not None) / len(comparison):.1f}%

"""
    
    report += f"""
## Individual Object Results

| Object | Generated | Previous | Improvement | Status |
|--------|-----------|----------|-------------|---------|
"""
    
    for comp in sorted(comparison, key=lambda x: x.get('improvement', 0), reverse=True):
        status = "✅" if comp['generated_success'] else "❌"
        improvement = f"+{comp['improvement']:.1f}%" if comp['improvement'] > 0 else f"{comp['improvement']:.1f}%"
        
        report += f"| {comp['object_name']} | {comp['generated_working']}/{comp['generated_total']} ({comp['generated_rate']:.1f}%) | {comp['previous_working']}/{comp['previous_total']} ({comp['previous_rate']:.1f}%) | {improvement} | {status} |\n"
    
    report += f"""

## Detailed Results by Object

"""
    
    for result in generated_results:
        if result.get('success', False):
            report += f"""
### {result['object_name'].title()}
- **Success Rate**: {result['success_rate']:.1f}% ({result['working_methods']}/{result['total_methods']} methods)
- **Working Methods**: {len(result.get('working_method_names', []))}
- **Failing Methods**: {len(result.get('failing_method_names', []))}
"""
        else:
            report += f"""
### {result['object_name'].title()} ❌
- **Status**: Failed to run
- **Error**: {result.get('error', 'Unknown error')}
"""
    
    report += f"""

## Next Steps

1. **High Priority**: Fix failing test scripts 
2. **Medium Priority**: Improve argument patterns for methods with low success rates
3. **Documentation**: Use successful test patterns to generate comprehensive API docs
4. **Integration**: Incorporate these tests into the main Groggy test suite

---
*Generated by test_results_aggregator.py*
"""
    
    return report

def main():
    """Run all generated test scripts and create comparison report"""
    print("🚀 Running Generated Test Scripts Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Find all test scripts
    test_dir = "documentation/testing/generated_tests"
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    test_scripts = [
        os.path.join(test_dir, f) for f in os.listdir(test_dir) 
        if f.startswith('test_') and f.endswith('.py')
    ]
    
    if not test_scripts:
        print(f"❌ No test scripts found in {test_dir}")
        return
    
    print(f"📋 Found {len(test_scripts)} test scripts")
    
    # Run all test scripts
    generated_results = []
    for script in sorted(test_scripts):
        result = run_test_script(script)
        generated_results.append(result)
    
    # Load previous results
    print(f"\n📊 Loading previous comprehensive API test results...")
    previous_results = load_previous_csv_results()
    
    # Compare results
    print(f"🔍 Comparing results...")
    comparison = compare_results(generated_results, previous_results)
    
    # Generate report
    print(f"📝 Generating comparison report...")
    report = generate_report(generated_results, previous_results, comparison)
    
    # Save report
    report_file = "documentation/testing/GENERATED_VS_DYNAMIC_TESTING_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_file}")
    
    # Print summary to console
    successful_scripts = [r for r in generated_results if r.get('success', False)]
    print(f"\n# Summary")
    print(f"- **Generated Scripts**: {len(successful_scripts)}/{len(generated_results)} successful")
    
    if previous_results and comparison:
        avg_improvement = sum(r['improvement'] for r in comparison if r['improvement'] is not None) / len(comparison)
        print(f"- **Average Improvement**: {avg_improvement:.1f}%")
        
        best_improvement = max(comparison, key=lambda x: x.get('improvement', 0))
        print(f"- **Best Improvement**: {best_improvement['object_name']} (+{best_improvement['improvement']:.1f}%)")
    
    return generated_results, comparison

if __name__ == "__main__":
    results, comparison = main()