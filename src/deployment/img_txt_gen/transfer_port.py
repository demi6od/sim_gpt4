from flask import request
from flask import Flask
from flask_cors import CORS
import requests

node_url = "http://10.198.34.146:12015/gpt4"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

@app.route('/gpt4', methods=['POST'])
def gpt4_server():
    r = requests.post(node_url, json=request.json)
    return r.json()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=20101)
