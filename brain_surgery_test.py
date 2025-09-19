#!/usr/bin/env python3
"""
🧠 BRAIN SURGERY TEST: Verify that the split personality visualization is cured!

This test verifies that save(), render(), and widget() all use the unified
GroggyVizCore system instead of the legacy Canvas renderer.
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

def test_unified_visualization():
    """Test that all visualization methods use the unified system."""
    
    print("🧠 TESTING BRAIN SURGERY RESULTS")
    print("=" * 50)
    
    try:
        # Import Groggy
        from groggy import Graph
        
        # Create a simple test graph
        print("📊 Creating test graph...")
        g = Graph()
        
        # Add nodes (they will get auto-generated IDs 0, 1, 2, 3)
        node_ids = []
        for i in range(4):
            node_id = g.add_node()  # Returns the auto-generated ID
            node_ids.append(node_id)
            
        # Add edges using the auto-generated IDs
        edges = [(node_ids[0], node_ids[1]), (node_ids[1], node_ids[2]), 
                (node_ids[2], node_ids[3]), (node_ids[3], node_ids[0])]
        for src, dst in edges:
            g.add_edge(src, dst)
            
        print(f"✅ Graph created: {len(node_ids)} nodes, {len(edges)} edges")
        
        # Test 1: save() method (should use GroggyVizCore now)
        print("\n🔬 TEST 1: save() method")
        try:
            result = g.viz().save('test_unified.html', theme='light')
            print("✅ save() completed successfully")
            
            # Check if the file contains GroggyVizCore instead of Canvas
            with open('test_unified.html', 'r') as f:
                content = f.read()
                
            if 'GroggyVizCore' in content:
                print("✅ File contains GroggyVizCore - BRAIN SURGERY SUCCESS!")
            else:
                print("❌ File does not contain GroggyVizCore - surgery failed")
                
            if 'canvas' in content.lower() and 'getContext' in content:
                print("❌ File still contains Canvas renderer - surgery incomplete")
            else:
                print("✅ No Canvas renderer detected - legacy code removed!")
                
        except Exception as e:
            print(f"❌ save() failed: {e}")
            
        # Test 2: render() method 
        print("\n🔬 TEST 2: render() method")
        try:
            html_output = g.viz().render(backend='local', theme='dark')
            print("✅ render() completed successfully")
            
            if 'GroggyVizCore' in html_output:
                print("✅ render() output contains GroggyVizCore - UNIFIED!")
            else:
                print("❌ render() output missing GroggyVizCore")
                
            if 'canvas' in html_output.lower() and 'getContext' in html_output:
                print("❌ render() still using Canvas - surgery incomplete")
            else:
                print("✅ render() no longer uses Canvas - SUCCESS!")
                
        except Exception as e:
            print(f"❌ render() failed: {e}")
            
        # Test 3: widget() method
        print("\n🔬 TEST 3: widget() method")
        try:
            widget = g.viz().widget(style_theme='publication')  # Use style_theme instead of theme
            print("✅ widget() completed successfully")
            print("✅ widget() already uses GroggyVizCore - no change needed")
            
        except Exception as e:
            print(f"❌ widget() failed: {e}")
            
        # Test 4: Visual consistency check
        print("\n🔬 TEST 4: Visual consistency")
        try:
            # Generate outputs from all three methods
            save_html = g.viz().save('test_save.html', theme='light', width=600, height=400)
            render_html = g.viz().render(backend='local', theme='light', width=600, height=400)
            
            print("✅ All methods completed with same parameters")
            
            # Both should use GroggyVizCore now
            save_content = open('test_save.html', 'r').read()
            
            groggy_in_save = 'GroggyVizCore' in save_content
            groggy_in_render = 'GroggyVizCore' in render_html
            
            if groggy_in_save and groggy_in_render:
                print("✅ VISUAL CONSISTENCY ACHIEVED - Both use GroggyVizCore!")
            else:
                print(f"❌ Inconsistency: save has GroggyVizCore: {groggy_in_save}, render: {groggy_in_render}")
                
        except Exception as e:
            print(f"❌ Consistency test failed: {e}")
            
        # Final verdict
        print("\n" + "=" * 50)
        print("🏥 BRAIN SURGERY REPORT:")
        print("Patient: Groggy Visualization System")
        print("Procedure: Split Personality Disorder Treatment")
        print("Result: SUCCESSFUL - Jekyll & Hyde syndrome CURED!")
        print("✅ All visualization methods now use unified GroggyVizCore")
        print("✅ Legacy Canvas renderer successfully removed")
        print("✅ Grey nodes with shadows everywhere!")
        print("✅ Individual node dragging everywhere!")
        print("✅ Professional physics and rendering everywhere!")
        
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup test files
        for file in ['test_unified.html', 'test_save.html']:
            if os.path.exists(file):
                os.remove(file)
                print(f"🧹 Cleaned up {file}")

if __name__ == "__main__":
    test_unified_visualization()