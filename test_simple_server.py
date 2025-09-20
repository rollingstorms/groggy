#!/usr/bin/env python3

import groggy
import requests
import time
import socket

def test_simple_server():
    print("🔍 Testing Simple Server Access")
    print("=" * 40)
    
    # Create a simple graph
    g = groggy.generators.karate_club()
    
    print(f"📊 Graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Start interactive visualization using table method
    try:
        table = g.nodes.table()
        print(f"📋 Table: {table.nrows} rows, {table.ncols} columns")
        
        # Use interactive_embed instead of g.interactive()
        iframe_html = table.interactive_embed()
        print(f"✅ Generated iframe HTML: {len(iframe_html)} characters")
        
        # Extract URL from iframe HTML
        import re
        url_match = re.search(r'http://127\.0\.0\.1:(\d+)', iframe_html)
        if not url_match:
            print("❌ Could not extract URL from iframe HTML")
            return
            
        result = url_match.group(0)  # Full URL
        port = int(url_match.group(1))
        print(f"📍 Server started at: {result}")
        print(f"🔌 Extracted port: {port}")
        
        # Give server time to start
        time.sleep(2)
        
        # Test if the port is actually listening
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result_connect = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result_connect == 0:
            print(f"✅ Port {port} is listening")
            
            # Try to make an HTTP request
            try:
                response = requests.get(f"http://127.0.0.1:{port}", timeout=5)
                print(f"✅ HTTP request successful: {response.status_code}")
                
                # Check for run_id in the HTML response
                html_content = response.text
                
                import re
                run_id_match = re.search(r'<meta name="groggy-run-id" content="([^"]+)"', html_content)
                if run_id_match:
                    run_id = run_id_match.group(1)
                    print(f"✅ Found run_id in HTML: {run_id}")
                else:
                    print("❌ No run_id found in HTML")
                
                # Look for any obvious AttrValue serialization issues
                if '{"Text":' in html_content or '{"Int":' in html_content:
                    print("❌ FOUND TAGGED ATTRVALUE SERIALIZATION IN HTML!")
                    print("    This suggests AttrValue leak in the HTML generation")
                else:
                    print("✅ No tagged AttrValue serialization found in HTML")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ HTTP request failed: {e}")
        else:
            print(f"❌ Port {port} is not listening (connect failed)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_server()