import json
from PIL import Image
import base64
import random
import shutil
import requests
import re
import sys
import os
import time
import math
from pprint import pprint

import pandas as pd
from tqdm import tqdm

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '../../..')
sys.path.append(PROJECT_DIR)

from src.utils.utils import data_split, read_file, write_file, remove_whilespace, cut_sentences, read_dir, qt_equal, read_df
from src.process.img_txt_gen.Processor import Processor
from src.utils.Metric import Metric

project_dir = '../../..'
data_pre_dir = 'data/img_txt_gen'

user_token = '<用户>'
sys_token = '<系统>'

g_n_bad = 0
g_n_none_anno = 0

metric = Metric()
max_sample = int(1e10)


def proc_wukong(dataset):
    data_dir = os.path.join(project_dir, data_pre_dir, dataset)
    in_file = os.path.join(data_dir, 'wukong_100m_0.csv')
    data = read_df(in_file)
    data = data[:max_sample]
    new_samples = []
    for id, sample in enumerate(tqdm(data)):
        new_sample = {
            'id': id,
            'caption': sample['caption'],
            'dataset': dataset,
        }

        image_url = sample['url']
        img_data = requests.get(image_url).content
        out_file = os.path.join(data_dir, 'images', f'{id}.jpg')
        with open(out_file, 'wb') as handler:
            handler.write(img_data)
        new_samples.append(new_sample)
        print(new_sample)
        time.sleep(1)
    write_file(data_dir, 'captions.json', new_samples)

def proc_coco(dataset, copy=False):
    data_dir = os.path.join(project_dir, data_pre_dir, dataset)
    in_dir = os.path.join(data_dir, 'label')
    files = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    data = []
    for file in files:
        in_file = os.path.join(in_dir, file)
        if 'translate' in file:
            sep = ' '
            df = pd.read_csv(in_file, sep=sep, lineterminator='\n', names=['id', 'caption'], on_bad_lines='skip')
        else:
            sep = '\t'
            df = pd.read_csv(in_file, sep=sep, lineterminator='\n', names=['id', 'caption'])
        data += df.to_dict('records')

    new_samples = []
    img_ids = []
    for sample in tqdm(data):
        img_id = sample['id'].split('#')[0]
        sample['id'] = img_id
        img_ids.append(img_id)
        sample['dataset'] = dataset

        if copy:
            src = os.path.join(data_dir, 'val2014', f'{img_id}.jpg')
            dst = os.path.join(data_dir, 'images', f'{img_id}.jpg')
            shutil.copyfile(src, dst)
        new_samples.append(sample)
    write_file(data_dir, 'captions.json', new_samples)
    # print(len(img_ids))
    # img_ids = list(set(img_ids))
    # print(len(img_ids))

def pre_proc():
    data_dir = os.path.join(project_dir, data_pre_dir)
    in_file = os.path.join(data_dir, 'captions.json')
    data = read_file(in_file)

    data_split(data, data_dir, train_ratio=0.9, val_ratio=0.05, is_shuffle=True)

def merge_all(datasets):
    data_dir = os.path.join(project_dir, data_pre_dir)
    data = []
    max_cap_len = 0
    for dataset in datasets:
        in_file = os.path.join(data_dir, dataset, 'captions.json')
        samples = read_file(in_file)
        for sample in samples:
            caption = sample['caption']
            max_cap_len = max(max_cap_len, len(caption))
            if len(caption) <= 3:
                print(sample)
        data += samples
        print(f'[+] {dataset}: {len(samples)}')

    print(f'[+] max_cap_len = {max_cap_len}')
    write_file(data_dir, 'captions.json', data)

def main():
    # proc_wukong(dataset='wukong')
    # proc_coco(dataset='coco')

    datasets = ['coco']
    # merge_all(datasets)

    pre_proc()
    print('[+] Finish')


if __name__ == '__main__':
    main()
