from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建HTTP服务器
server_address = ('', 8000)
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
print('Starting server on port 8000...')
httpd.serve_forever() 