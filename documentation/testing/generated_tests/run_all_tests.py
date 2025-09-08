#!/usr/bin/env python3
"""
Master test runner for all generated Groggy object tests
Generated on: 2025-09-07 21:42:37

This script runs all individual object test scripts and aggregates results.
"""

import os
import subprocess
import sys
from datetime import datetime

def run_test_script(script_path):
    """Run a single test script and capture output"""
    print(f"\n🧪 Running {os.path.basename(script_path)}...")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✅ Completed successfully")
            return True, result.stdout
        else:
            print(f"  ❌ Failed with return code {result.returncode}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout after 5 minutes")
        return False, "Timeout"
    except Exception as e:
        print(f"  💥 Error running script: {e}")
        return False, str(e)

def main():
    """Run all generated test scripts"""
    print("🚀 Master Test Runner for Groggy Object Tests")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    test_scripts = [
        "documentation/testing/generated_tests/test_graph.py",
        "documentation/testing/generated_tests/test_nodestable.py",
        "documentation/testing/generated_tests/test_edgestable.py",
        "documentation/testing/generated_tests/test_grapharray.py",
        "documentation/testing/generated_tests/test_graphmatrix.py",
        "documentation/testing/generated_tests/test_graphtable.py",
        "documentation/testing/generated_tests/test_subgraph.py",
        "documentation/testing/generated_tests/test_neighborhoodresult.py",
        "documentation/testing/generated_tests/test_basetable.py",
    ]
    
    results = []
    
    for script in test_scripts:
        if os.path.exists(script):
            success, output = run_test_script(script)
            results.append({'script': script, 'success': success, 'output': output})
        else:
            print(f"⚠️ Script not found: {script}")
            results.append({'script': script, 'success': False, 'output': 'File not found'})
    
    # Generate summary
    print(f"\n# Master Test Summary")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n**Overall Results**: {len(successful)}/{len(results)} test scripts completed successfully")
    
    if successful:
        print(f"\n**Successful Scripts:**")
        for r in successful:
            print(f"  ✅ {os.path.basename(r['script'])}")
    
    if failed:
        print(f"\n**Failed Scripts:**")
        for r in failed:
            print(f"  ❌ {os.path.basename(r['script'])}: {r['output'][:100]}...")
    
    # Save detailed results
    with open('master_test_results.txt', 'w') as f:
        f.write(f"Master Test Results - {datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
        
        for r in results:
            f.write(f"Script: {r['script']}\n")
            f.write(f"Success: {r['success']}\n")
            f.write(f"Output:\n{r['output']}\n")
            f.write("-" * 30 + "\n\n")
    
    print(f"\n📄 Detailed results saved to: master_test_results.txt")
    return results

if __name__ == "__main__":
    results = main()
