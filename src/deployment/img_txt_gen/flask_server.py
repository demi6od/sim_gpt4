import sys
import os
import traceback

from flask import request
from flask import Flask
from flask_cors import CORS
import logging

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '../../..')
sys.path.append(PROJECT_DIR)

from src.gpt4.Gpt4Infer import Gpt4Infer
from src.utils.utils import log_config
from src.deployment.server_api import nlg_api

logger = logging.getLogger('gpt4_server')

PORT = 12015
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)


@app.route('/gpt4', methods=['POST'])
def gpt4_server():
    rsp = nlg_api(nlg_infer, request.json)
    return rsp

def main():
    app.run(host='0.0.0.0', port=PORT)


sys.argv = ['', '../../params/img_txt_gen/itg_bloomz_infer.json']
args = {
    'project_dir': PROJECT_DIR,
}
nlg_infer = Gpt4Infer(args)
log_config(nlg_infer.args, 'server')
logger.info(nlg_infer.args)

if __name__ == '__main__':
    main()
