import traceback
import logging
import time
import json
import numpy as np

logger = logging.getLogger('nlg_server')

def nlg_api(nlg_infer, req, batch=False):
    try:
        start_time = time.time()
        if batch:
            out = nlg_infer.batch_run(req)
        else:
            out = nlg_infer.run(req)
        tot_time = time.time() - start_time
        rsp = {
            'response': out,
            'status': 'OK',
            'cost': tot_time,
        }
        return rsp
    except Exception as e:
        message = str(traceback.format_exc())
        logger.error(message)
        raise Exception(message)
