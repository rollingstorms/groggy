#!/usr/bin/env python3
"""
Simple development server for the streaming visualization template.

Usage:
    python streaming_dev_server.py

Opens browser to the streaming template for rapid CSS prototyping.
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def main():
    PORT = 8080
    DIRECTORY = "streaming_prototype"

    # Change to the prototype directory
    os.chdir(DIRECTORY)

    # Create HTTP server
    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 Streaming Template Server")
        print(f"   Server running at: http://localhost:{PORT}")
        print(f"   Template: http://localhost:{PORT}/streaming_template.html")
        print(f"   CSS Playground: http://localhost:{PORT}/css_playground.html")
        print(f"   Press Ctrl+C to stop")

        # Open browser
        webbrowser.open(f'http://localhost:{PORT}/streaming_template.html')

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n   Server stopped")

if __name__ == "__main__":
    main()