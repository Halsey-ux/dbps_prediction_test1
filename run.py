from waitress import serve
from app import app
import logging

logging.basicConfig(level=logging.INFO)
 
if __name__ == '__main__':
    print("Starting server on http://localhost:8000")
    serve(app, host='0.0.0.0', port=8000) 