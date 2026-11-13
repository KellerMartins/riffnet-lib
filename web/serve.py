"""
Simple static file server with correct MIME types for WASM and ES modules.
Sends no-cache headers on every response so mobile browsers never serve stale files.
Usage: python serve.py [port]
"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8081

# Serve this script's own directory regardless of the caller's cwd, so
# `python web/serve.py` (from the repo root, per README.md) and
# `python serve.py` (from inside web/) both serve the same files.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    # Extend the default type map with types browsers strictly require
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.js':   'application/javascript',
        '.mjs':  'application/javascript',
        '':      'application/octet-stream',
    }

    def end_headers(self):
        # Prevent all caching
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, fmt, *args):
        # Suppress noisy access log; keep only HTTP errors
        try:
            if int(args[1]) >= 400:
                super().log_message(fmt, *args)
        except (IndexError, ValueError):
            pass  # log_error calls with non-numeric args — ignore

if __name__ == '__main__':
    with http.server.HTTPServer(('', PORT), Handler) as httpd:
        print(f'Serving web/ on http://localhost:{PORT}  (Ctrl-C to stop)')
        httpd.serve_forever()
