import os
from huggingface_hub import snapshot_download

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '../..')

model_dir = os.path.join(PROJECT_DIR, 'model/cache')
snapshot_download(repo_id='bigscience/bloomz-1b1', ignore_regex=['*.h5', '*.ot', '*.msgpack', '*.safetensors'], cache_dir=model_dir)

print('[+] Finish')