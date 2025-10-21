#!/usr/bin/env python3
"""
Enhanced development server for Groggy template prototyping with live reload

Usage:
    python dev_server.py [port]

Default port: 8000

Features:
- Serves template files
- Live reload on file changes
- WebSocket-based refresh
- Custom MIME types for CSS
"""

import http.server
import socketserver
import sys
import webbrowser
import os
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import hashlib

class LiveReloadHandler(http.server.SimpleHTTPRequestHandler):
    """Enhanced HTTP handler with live reload capabilities"""
    
    def __init__(self, *args, file_watcher=None, **kwargs):
        self.file_watcher = file_watcher
        super().__init__(*args, **kwargs)
    
    def end_headers(self):
        """Add CORS headers for local development"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        """Handle GET requests with live reload injection"""
        if self.path == '/livereload':
            self.handle_livereload_check()
            return
        elif self.path.endswith('.html'):
            self.serve_html_with_livereload()
            return
        else:
            super().do_GET()
    
    def handle_livereload_check(self):
        """Handle live reload polling endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        current_hash = self.file_watcher.get_files_hash() if self.file_watcher else ""
        response = json.dumps({"hash": current_hash})
        self.wfile.write(response.encode())
    
    def serve_html_with_livereload(self):
        """Serve HTML files with live reload script injected"""
        try:
            file_path = self.translate_path(self.path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Inject live reload script before closing </body> tag
            livereload_script = '''
    <script>
        // Live reload functionality
        let lastHash = '';
        
        function checkForChanges() {
            fetch('/livereload')
                .then(response => response.json())
                .then(data => {
                    if (lastHash && lastHash !== data.hash) {
                        console.log('📁 Files changed, reloading...');
                        window.location.reload();
                    }
                    lastHash = data.hash;
                })
                .catch(error => {
                    console.log('Live reload check failed:', error);
                });
        }
        
        // Check for changes every 1 second
        setInterval(checkForChanges, 1000);
        checkForChanges(); // Initial check
    </script>
</body>'''
            
            if '</body>' in content:
                content = content.replace('</body>', livereload_script)
            else:
                # If no </body> tag, append at end
                content += livereload_script
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content.encode())))
            self.end_headers()
            self.wfile.write(content.encode())
            
        except Exception as e:
            self.send_error(404, f"File not found: {e}")

class FileWatcher:
    """Watch files for changes"""
    
    def __init__(self, watch_dir):
        self.watch_dir = Path(watch_dir)
        self.last_hash = ""
    
    def get_files_hash(self):
        """Get hash of all watched files"""
        hash_content = ""
        
        # Watch CSS files, HTML files, and JS files
        patterns = ["*.css", "*.html", "*.js"]
        
        for pattern in patterns:
            for file_path in self.watch_dir.rglob(pattern):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        hash_content += f"{file_path}:{stat.st_mtime}:{stat.st_size};"
                    except:
                        pass
        
        return hashlib.md5(hash_content.encode()).hexdigest()

def create_handler(file_watcher):
    """Create handler with file watcher"""
    class Handler(LiveReloadHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, file_watcher=file_watcher, **kwargs)
    return Handler

def main():
    # Change to template_prototype directory
    template_dir = Path("template_prototype")
    if not template_dir.exists():
        print("❌ template_prototype directory not found!")
        print("   Run 'python template_generator.py' first")
        return
    
    os.chdir(template_dir)
    
    # Get port from command line or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    # Create file watcher
    file_watcher = FileWatcher(".")
    
    # Create server with live reload handler
    handler = create_handler(file_watcher)
    httpd = socketserver.TCPServer(("", port), handler)
    
    print(f"🚀 Groggy Template Server (with Live Reload)")
    print(f"   📡 Serving at http://localhost:{port}")
    print(f"   📁 Directory: {Path.cwd()}")
    print(f"   🎨 Main template: http://localhost:{port}/template.html")
    print(f"   🎭 Theme selector: http://localhost:{port}/theme_selector.html")
    print(f"   🎪 CSS playground: http://localhost:{port}/css_playground.html")
    print(f"   🔄 Live reload: Enabled (watching CSS, HTML, JS files)")
    print(f"   ⏹️  Press Ctrl+C to stop")
    
    # Try to open browser
    try:
        webbrowser.open(f"http://localhost:{port}/template.html")
    except:
        pass
    
    # Start server
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    main()
