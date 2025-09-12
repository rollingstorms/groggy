#!/usr/bin/env python3
"""Debug script to check server connectivity with manual curl tests"""

import groggy
import time
import subprocess
import sys

def test_server_with_curl():
    print("🧪 Starting server and testing with curl...")
    
    # Create a small graph
    g = groggy.Graph()
    for i in range(3):
        g.add_node(i, name=f"Node_{i}")
    
    # Start server
    url = g.nodes.table().interactive()
    print(f"🚀 Server URL: {url}")
    
    # Extract port from URL
    port = url.split(':')[-1]
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Test IPv4 vs IPv6
    print(f"\n🧪 Testing IPv4 (127.0.0.1:{port})...")
    try:
        result = subprocess.run(['curl', '-v', f'http://127.0.0.1:{port}/'], 
                              capture_output=True, text=True, timeout=5)
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT: {result.stdout[:500]}...")
        print(f"STDERR: {result.stderr[:500]}...")
    except subprocess.TimeoutExpired:
        print("❌ IPv4 request timed out")
    except Exception as e:
        print(f"❌ IPv4 error: {e}")
    
    print(f"\n🧪 Testing IPv6 (localhost:{port})...")
    try:
        result = subprocess.run(['curl', '-v', f'http://localhost:{port}/'], 
                              capture_output=True, text=True, timeout=5)
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT: {result.stdout[:500]}...")
        print(f"STDERR: {result.stderr[:500]}...")
    except subprocess.TimeoutExpired:
        print("❌ IPv6 request timed out")
    except Exception as e:
        print(f"❌ IPv6 error: {e}")
    
    # Also check what's listening
    print(f"\n🔍 Checking what's listening on port {port}...")
    try:
        result = subprocess.run(['lsof', '-i', f':{port}'], 
                              capture_output=True, text=True, timeout=5)
        print(f"lsof output: {result.stdout}")
    except Exception as e:
        print(f"❌ lsof error: {e}")

if __name__ == "__main__":
    test_server_with_curl()