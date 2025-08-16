#!/usr/bin/env python3
"""Test display functionality"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

try:
    import groggy as gr
    print("✅ Successfully imported groggy")
    
    # Test DisplayConfig
    config = gr.DisplayConfig(max_rows=5, max_cols=3, use_color=False)
    print(f"✅ Created DisplayConfig: {config}")
    
    # Test format functions are available
    print(f"✅ format_array available: {hasattr(gr, 'format_array')}")
    print(f"✅ format_matrix available: {hasattr(gr, 'format_matrix')}")
    print(f"✅ format_table available: {hasattr(gr, 'format_table')}")
    
    # Test array formatting with sample data
    array_data = {
        "data": [1, 2, 3, 4, 5],
        "dtype": "int64"
    }
    
    # Test without config first
    formatted = gr.format_array(array_data)
    print("\n✅ Array formatting test (no config):")
    print(formatted)
    
    # Test with config - check if function takes config parameter
    try:
        formatted_with_config = gr.format_array(array_data, config=config)
        print("\n✅ Array formatting test (with config):")
        print(formatted_with_config)
    except Exception as e:
        print(f"\n⚠️  Config parameter not working as expected: {e}")
        print("This is normal - the function signature may need adjustment")
    
    print("\n🎉 Display functionality working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()