# -*- coding:utf-8 -*-
import gradio
import requests
import time
import json
import numpy as np

gpt4_url = 'http://10.198.34.48:12015/gpt4'

def gpt4_server(sample, task, timeout=100, print_time=True):
    start_time = time.time()
    r = requests.post(gpt4_url, json=sample, timeout=timeout)
    response = r.json()
    if response['status'] != 'OK' or 'response' not in response:
        err_msg = f'[-] {task} Server error: {response["status"]}'
        print(err_msg)
        raise Exception(err_msg)
    resp = response['response']
    if print_time:
        print(f'[+] %s time: %.3fs' % (task, time.time() - start_time))
    return resp

def gpt4_resp(query, img):
    img_json = json.dumps(np.array(img).tolist())
    sample = {
        'dialog_his': '',
        'knows': [''],
        'query': query,
        'img_json': img_json,
    }
    out = gpt4_server(sample, 'gpt4')
    return out['pred']

def main():
    demo = gradio.Interface(
        inputs=[gradio.Textbox(lines=1, label='query'), gradio.Image(label='image', type='pil', source='upload')],
        outputs="textbox",
        model_type='pyfunc',
        fn=gpt4_resp,
    )

    demo.launch(inline=True, share=True)
    time.sleep(1e4)
    print('[+] Finish!')


if __name__ == '__main__':
    main()
