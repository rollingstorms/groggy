#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy

def test_path_result():
    print("🧪 Testing PathResult implementation...")
    
    # Test basic PathResult creation
    print("\n=== Testing PathResult Creation ===")
    
    # Create a PathResult manually (testing the FFI wrapper)
    try:
        path_result = groggy._groggy.PathResult()
        print("ERROR: PathResult() should require arguments")
    except Exception as e:
        print(f"✅ PathResult() correctly requires arguments: {type(e).__name__}")
    
    # Test PathResult properties
    print("\n=== Testing PathResult Properties ===")
    
    # For now, let's test if the class is available
    print(f"✅ PathResult class available: {groggy._groggy.PathResult}")
    print(f"✅ PathResult type: {type(groggy._groggy.PathResult)}")
    
    # Test that we can import it
    from groggy._groggy import PathResult
    print(f"✅ PathResult imported successfully: {PathResult}")
    
    print("\n🎉 PathResult basic infrastructure is working!")
    print("📝 Note: Need to implement direct Graph.bfs_fast() methods to create PathResult instances")

if __name__ == "__main__":
    test_path_result()