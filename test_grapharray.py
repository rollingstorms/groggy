#!/usr/bin/env python3

"""
Test GraphArray functionality - verify renamed API works correctly
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

def test_grapharray_basic():
    """Test basic GraphArray functionality"""
    print("🧪 Testing GraphArray basic functionality...")
    
    # Create GraphArray directly
    try:
        ages = groggy.GraphArray([25, 30, 35, 40, 45])
        print(f"✅ GraphArray created: {ages}")
        
        # Test statistical methods
        mean = ages.mean()
        std = ages.std()
        minimum = ages.min()
        maximum = ages.max()
        
        print(f"✅ Statistical methods work:")
        print(f"   Mean: {mean}")
        print(f"   Std: {std}")
        print(f"   Min: {minimum}")
        print(f"   Max: {maximum}")
        
        # Test list compatibility
        print(f"✅ List compatibility:")
        print(f"   Length: {len(ages)}")
        print(f"   First: {ages[0]}")
        print(f"   Last: {ages[-1]}")
        print(f"   Iteration: {list(ages)}")
        
        return True
        
    except Exception as e:
        print(f"❌ GraphArray test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_grapharray_basic()
    if success:
        print("\n🎉 GraphArray API renaming successful!")
    else:
        print("\n❌ GraphArray API renaming failed!")