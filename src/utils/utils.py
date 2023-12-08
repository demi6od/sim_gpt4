import re
import os
import sys
import json
import logging
import random
import datetime
import torch
from random import shuffle
import pandas as pd
from thefuzz import fuzz

logger = logging.getLogger('utils')

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def read_args(args=None):
    # Read arguments
    if len(sys.argv) < 2:
        logging.error('[-] Please set params file as first argument')
    with open(sys.argv[1]) as f:
        file_args = json.load(f)
        file_args = DotDict(file_args)

    if args is not None:
        for name, val in args.items():
            file_args[name] = val
    return file_args

def parse_args(args):
    if args.project_dir is None:
        args.project_dir = '../'

    if args.data_dir is None:
        args.data_dir = os.path.join(args.project_dir, 'data', args.task)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.project_dir, 'output', args.task)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_dir is None:
        args.model_dir = args.output_dir

    if 'gpt' in args.lang_model_name:
        args.model_type = 'gpt'
    elif 't5' in args.lang_model_name:
        args.model_type = 't5'
    elif 'bloomz' in args.lang_model_name:
        args.model_type = 'bloomz'
    elif 'mt0' in args.lang_model_name:
        args.model_type = 'mt0'

    if args.load_model is None:
        if args.run_type == 'train':
            args.load_model = False
        else:
            args.load_model = True
    if args.do_sample is None:
        args.do_sample = True
    if args.top_p is None:
        args.top_p = 0.5
    if args.print_step is None:
        args.print_step = int(5e4)
    if args.clip_grad is None:
        args.clip_grad = True

    if args.parallel is None:
        args.parallel = False

    if args.metric is None:
        args.metric = 'bleu'
    if args.temp is None:
        args.temp = 1
    if args.model_lr is None:
        args.model_lr = 1e-5

    if args.seed is None:
        args.seed = 3407
    return args

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
        else:
            # logger.info(f'[+] no_grad param: {name}')
            pass

def log_config(args, t):
    now = datetime.datetime.now()
    args.now = str(now).rsplit('.', 1)[0].replace(':', '_').replace(' ', '_')
    log_name = t + '_' + args.now + '.log'

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_name), 'w'),
            logging.StreamHandler()
        ]
    )

def data_split(tot_data, out_dir, train_ratio=0.95, val_ratio=0.025, is_shuffle=False, save_file=True):
    if is_shuffle:
        shuffle(tot_data)

    tot_len = len(tot_data)
    train_len = int(train_ratio * tot_len)
    val_len = int(val_ratio * tot_len)

    data = {
        'train': tot_data[:train_len],
        'dev': tot_data[train_len:(train_len + val_len)],
        'test': tot_data[(train_len + val_len):]
    }

    if save_file:
        for data_t in ['train', 'dev', 'test']:
            write_file(out_dir, data_t + '.json', data[data_t])
    return data

def remove_whilespace(text, space='\s'):
    text = str(text)
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1}[' + space + ']+(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list, key=lambda i:len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i, new_i)
    return text

def read_dir(in_dir, file_type='json', multi_lines=False):
    files = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    data = []
    for file in files:
        in_file = os.path.join(in_dir, file)
        if file_type == 'json':
            data += read_file(in_file, multi_lines)
        elif file_type == 'df':
            data += read_df(in_file)
    return data

def read_file(in_file, multi_lines=False, errors='ignore'):
    if multi_lines:
        data = []
        for line in open(in_file, 'r', encoding='UTF-8', errors=errors):
            data.append(json.loads(line))
    else:
        with open(in_file, 'r', encoding='UTF-8', errors=errors) as f:
            data = json.load(f)
    print('[+] Read data:', len(data), in_file)
    return data

def write_file(out_dir, file_name, data):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, file_name)
    with open(out_file, 'w', encoding='UTF-8') as f:
        if isinstance(data, dict):
            data_len = len(list(data.values())[0])
        else:
            data_len = len(data)
        print('[+] Save data ' + str(data_len) + ' at', out_file)
        json.dump(data, f, indent=4, ensure_ascii=False)

def cut_sentences(para, lang='cn', drop_empty_line=True, strip=True, deduplicate=False, drop_newline=True):
    '''cut_sentences

    :param para: 输入文本
    :param drop_empty_line: 是否丢弃空行
    :param strip: 是否对每一句话做一次strip
    :param deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句
    :return: sentences: list of str
    '''
    if deduplicate:
        para = re.sub(r"([。！？\!\?])\1+", r"\1", para)

    if lang == 'en':
        from nltk import sent_tokenize
        sents = sent_tokenize(para)
        if strip:
            sents = [x.strip() for x in sents]
        if drop_empty_line:
            sents = [x for x in sents if len(x.strip()) > 0]
        return sents
    else:
        para = re.sub('([。！？\?!])([^”’)\]）】])', r"\1[sep]\2", para)  # 单字符断句符
        para = re.sub('(\.{3,})([^”’)\]）】….])', r"\1[sep]\2", para)  # 英文省略号
        para = re.sub('(\…+)([^”’)\]）】….])', r"\1[sep]\2", para)  # 中文省略号
        para = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2[sep]\3', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符[sep]放到双引号后，注意前面的几句都小心保留了双引号
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        if drop_newline:
            para = para.replace('\n', '')
        sentences = para.split("[sep]")
        if strip:
            sentences = [sent.strip() for sent in sentences]
        if drop_empty_line:
            sentences = [sent for sent in sentences if len(sent.strip()) > 0]
        return sentences

def rand_flip(prob):
    rand = random.random()
    if rand < prob:
        flip = True
    else:
        flip = False
    return flip

def qt_equal(q1, q2, txt1, txt2, fuzz_match=True):
    q1 = str(q1).replace(' ', '').rstrip(' ?？')
    q2 = str(q2).replace(' ', '').rstrip(' ?？')
    txt1 = str(txt1)
    txt2 = str(txt2)

    if fuzz_match:
        score_q = fuzz.ratio(q1, q2)
        score_t = fuzz.ratio(txt1, txt2)
        if score_q > 95 and score_t > 60:
            eq = True
        else:
            eq = False
    else:
        if q1 == q2 and txt1 == txt2:
            eq = True
        else:
            eq = False
    return eq

def read_df(in_file):
    if '~$' in in_file:
        return []

    if '.xls' in in_file:
        print(f'[+] Read excel from {in_file}')
        df_dic = pd.read_excel(in_file, sheet_name=None)
        data = []
        for key, df in df_dic.items():
            print(f'[+] Read sheet {key}: {len(df)}')
            df = df.fillna('')
            data += df.to_dict('records')
    elif '.csv' in in_file:
        print(f'[+] Read csv from {in_file}')
        df = pd.read_csv(in_file)
        df = df.fillna('')
        data = df.to_dict('records')
    else:
        print(f'[-] read_df file type error: {in_file}')
        return []
    return data

def write_df(out_dir, file_name, data):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, file_name)
    data_len = len(data)
    print('[+] Save data ' + str(data_len) + ' at', out_file)
    data.to_excel(out_file)
