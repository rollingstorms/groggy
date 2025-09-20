#!/usr/bin/env python3
"""
Comprehensive WebSocket data inspection test

This test specifically looks for the [object Object] issue by:
1. Connecting to the WebSocket directly
2. Inspecting the actual JSON data being sent
3. Testing both initial load and scroll updates
4. Flagging any AttrValue objects that aren't properly converted
"""

import sys
import os
import time
import json
import asyncio
import websockets
import threading
import requests

# Add python-groggy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

import groggy as gr

class WebSocketDataInspector:
    def __init__(self):
        self.messages_received = []
        self.object_object_detected = False
        self.raw_attr_values_detected = []
        self.connection = None

    async def connect_and_inspect(self, port):
        """Connect to WebSocket and inspect all messages"""
        uri = f"ws://127.0.0.1:{port}"

        try:
            print(f"🔌 Connecting to WebSocket: {uri}")
            async with websockets.connect(uri) as websocket:
                self.connection = websocket
                print("✅ WebSocket connected")

                # Send initial request (if needed)
                # Most servers send initial data automatically

                # Listen for messages
                message_count = 0
                timeout_count = 0
                max_timeouts = 3

                while timeout_count < max_timeouts:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        message_count += 1
                        timeout_count = 0  # Reset timeout counter

                        print(f"📨 Received message {message_count}: {len(message)} bytes")

                        # Inspect the message
                        self.inspect_message(message, message_count)

                        # Test scroll request after first message
                        if message_count == 1:
                            print("📜 Sending scroll request to test updates...")
                            scroll_request = {
                                "type": "ScrollRequest",
                                "offset": 10,
                                "window_size": 20
                            }
                            await websocket.send(json.dumps(scroll_request))

                    except asyncio.TimeoutError:
                        timeout_count += 1
                        print(f"⏰ Timeout {timeout_count}/{max_timeouts} waiting for WebSocket message")

                        if timeout_count >= max_timeouts:
                            print("⏹️  No more messages, ending inspection")
                            break

                print(f"📊 Total messages inspected: {message_count}")

        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False

        return True

    def inspect_message(self, message_text, message_num):
        """Inspect a WebSocket message for data quality issues"""
        print(f"\n🔍 Inspecting message {message_num}:")

        try:
            # Parse JSON
            data = json.loads(message_text)
            self.messages_received.append(data)

            print(f"  📋 Message type: {data.get('type', 'unknown')}")

            # Look for table data
            if 'window' in data:
                self.inspect_table_data(data['window'], message_num)
            elif 'data' in data:
                self.inspect_table_data(data['data'], message_num)
            elif 'rows' in data:
                self.inspect_rows(data['rows'], message_num)
            else:
                print(f"  ℹ️  No table data found in message")
                print(f"  🗝️  Keys: {list(data.keys())}")

        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            print(f"  📄 Raw message (first 200 chars): {message_text[:200]}")

    def inspect_table_data(self, table_data, message_num):
        """Inspect table data structure for issues"""
        print(f"  📊 Inspecting table data...")

        if not isinstance(table_data, dict):
            print(f"  ❌ Table data is not a dict: {type(table_data)}")
            return

        # Check for rows
        if 'rows' in table_data:
            rows = table_data['rows']
            print(f"  📏 Found {len(rows)} rows")
            self.inspect_rows(rows, message_num)
        else:
            print(f"  🗝️  Table data keys: {list(table_data.keys())}")

        # Check headers
        if 'headers' in table_data:
            headers = table_data['headers']
            print(f"  📑 Headers: {headers}")

    def inspect_rows(self, rows, message_num):
        """Inspect row data for [object Object] and AttrValue issues"""
        print(f"  🔬 Inspecting {len(rows)} rows for data quality...")

        issues_found = []
        sample_values = []

        for row_idx, row in enumerate(rows[:5]):  # Check first 5 rows
            if not isinstance(row, list):
                issues_found.append(f"Row {row_idx} is not a list: {type(row)}")
                continue

            for col_idx, cell_value in enumerate(row):
                # Collect sample values
                if row_idx < 3 and col_idx < 3:  # First 3x3 sample
                    sample_values.append(f"[{row_idx},{col_idx}]: {repr(cell_value)}")

                # Check for [object Object] issues
                cell_str = str(cell_value)
                if '[object Object]' in cell_str:
                    self.object_object_detected = True
                    issues_found.append(f"🚨 [object Object] detected at row {row_idx}, col {col_idx}")

                # Check for raw AttrValue structures (nested dicts with type info)
                if isinstance(cell_value, dict):
                    if self.looks_like_attr_value(cell_value):
                        self.raw_attr_values_detected.append({
                            'message': message_num,
                            'row': row_idx,
                            'col': col_idx,
                            'value': cell_value
                        })
                        issues_found.append(f"🔧 Raw AttrValue detected at row {row_idx}, col {col_idx}: {cell_value}")

                # Check for unexpected object types
                if isinstance(cell_value, (dict, list)) and not self.is_expected_complex_type(cell_value):
                    issues_found.append(f"🔍 Unexpected complex type at row {row_idx}, col {col_idx}: {type(cell_value)}")

        # Report findings
        print(f"  📋 Sample values:")
        for sample in sample_values:
            print(f"    {sample}")

        if issues_found:
            print(f"  🚨 ISSUES FOUND:")
            for issue in issues_found:
                print(f"    {issue}")
        else:
            print(f"  ✅ No data quality issues detected")

    def looks_like_attr_value(self, value):
        """Check if a dict looks like a raw AttrValue enum"""
        if not isinstance(value, dict):
            return False

        # AttrValue enums typically have variant names as keys
        attr_value_variants = {'Int', 'Float', 'Text', 'Bool', 'Null', 'CompactText',
                              'FloatVec', 'Bytes', 'NodeArray', 'EdgeArray', 'SubgraphRef'}

        return any(key in attr_value_variants for key in value.keys())

    def is_expected_complex_type(self, value):
        """Check if a complex type is expected (like arrays for certain columns)"""
        # Allow arrays for position data, etc.
        if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            return True
        return False

    def generate_report(self):
        """Generate final inspection report"""
        print("\n" + "="*60)
        print("📊 WEBSOCKET DATA INSPECTION REPORT")
        print("="*60)

        print(f"📨 Total messages received: {len(self.messages_received)}")

        if self.object_object_detected:
            print("🚨 [object Object] DETECTED - Issue is still present!")
        else:
            print("✅ No [object Object] detected")

        if self.raw_attr_values_detected:
            print(f"🔧 Raw AttrValue objects detected: {len(self.raw_attr_values_detected)}")
            print("   This indicates conversion is not happening correctly")
            for detection in self.raw_attr_values_detected[:3]:  # Show first 3
                print(f"   - Message {detection['message']}, Row {detection['row']}, Col {detection['col']}: {detection['value']}")
        else:
            print("✅ No raw AttrValue objects detected")

        # Analyze message types
        message_types = {}
        for msg in self.messages_received:
            msg_type = msg.get('type', 'unknown')
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        print(f"📋 Message types received:")
        for msg_type, count in message_types.items():
            print(f"   - {msg_type}: {count}")

        return len(self.raw_attr_values_detected) == 0 and not self.object_object_detected

async def run_websocket_inspection(port):
    """Run the WebSocket inspection"""
    inspector = WebSocketDataInspector()
    success = await inspector.connect_and_inspect(port)
    report_success = inspector.generate_report()
    return success and report_success

def test_websocket_data_quality():
    """Main test function"""
    print("🧪 WebSocket Data Quality Test")
    print("="*50)

    try:
        # Create graph and start server
        print("📊 Creating karate club graph...")
        g = gr.generators.karate_club()
        table = g.nodes.table()

        # Start visualization server
        print("🚀 Starting visualization server...")
        iframe = table.interactive_embed()

        # Extract port
        import re
        port_match = re.search(r'127\.0\.0\.1:(\d+)', iframe)
        if not port_match:
            print("❌ Could not extract port from iframe")
            return False

        port = int(port_match.group(1))
        print(f"✅ Server started on port {port}")

        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        time.sleep(2)

        # Test HTTP endpoint first
        print("🌐 Testing HTTP endpoint...")
        try:
            response = requests.get(f"http://127.0.0.1:{port}", timeout=5)
            if response.status_code == 200:
                print("✅ HTTP endpoint responding")
            else:
                print(f"⚠️  HTTP endpoint returned {response.status_code}")
        except Exception as e:
            print(f"❌ HTTP endpoint failed: {e}")
            return False

        # Run WebSocket inspection
        print("\n🔌 Starting WebSocket inspection...")
        success = asyncio.run(run_websocket_inspection(port))

        return success

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_websocket_data_quality()

    print("\n" + "="*50)
    if success:
        print("🎉 ALL TESTS PASSED - Data quality is good!")
    else:
        print("❌ TESTS FAILED - Data quality issues detected!")
        print("🔧 The [object Object] issue may still be present")
    print("="*50)

    exit(0 if success else 1)