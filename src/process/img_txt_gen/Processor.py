import os.path
from PIL import Image
import re
import json
import numpy as np
from src.utils.utils import remove_whilespace, DotDict
from src.utils.Metric import Metric

class Processor:
    def __init__(self, run_type, skip_ask, args):
        self.args = args
        self.run_type = run_type
        self.skip_ask = skip_ask

        self.max_his_len = 0
        self.max_know_len = 10
        if self.run_type == 'infer':
            self.max_user_len = 130
        else:
            self.max_user_len = 30

        self.max_img_len = 40
        if self.run_type == 'infer':
            self.max_post_len = 140
            self.max_resp_len = 0
        else:
            self.max_post_len = 40
            self.max_resp_len = 100

        self.his_token = '[对话历史]'
        self.user_token = '[当前对话]<用户>'
        self.resp_token = '[对话回复]<系统>'
        self.know_token = '[知识]'
        self.img_token = '[图片]'

        self.query = '能简单描述一下图片的内容吗？'

        self.anom_stat = {
            'n_anom_img': 0,
            'n_qst_type_resp': 0,
            'n_long_his': 0,
            'n_tot_his': 0,
            'n_long_know': 0,
            'n_tot_know': 0,
            'n_long_user': 0,
            'n_tot_user': 0,
            'n_long_resp': 0,
            'n_tot_resp': 0,
        }

        self.metric = Metric()

    def pre_proc(self, data):
        samples = self.convert_dataset_to_samples(data)
        return samples

    def convert_dataset_to_samples(self, data):
        itg_samples = []
        for sample in data:
            self.anom = False

            if 'img_json' in sample:
                src_img = Image.fromarray(np.array(json.loads(sample['img_json']), dtype='uint8'))
            elif 'img' in sample:
                src_img = sample['img']
            else:
                img_id = sample['id']
                dataset = sample['dataset']
                img_file = os.path.join(self.args.data_dir, dataset, 'images', f'{img_id}.jpg')
                try:
                    src_img = Image.open(img_file)
                except Exception as e:
                    self.anom_stat['n_anom_img'] += 1
                    continue

            if self.run_type == 'infer':
                dialog_his_raw = sample['dialog_his']
                query = sample['query']
                know = '||'.join(sample['knows'])
            else:
                dialog_his_raw = ''
                query = self.query
                know = ''
            dialog_his = self.his2str(dialog_his_raw)
            user_query_str = self.user2str(query)
            know_str = self.know2str(know)

            src_pre = dialog_his + know_str + self.img_token
            src_post =  user_query_str
            src_post = self.post2str(src_post)
            if 'caption' in sample:
                response = sample['caption']
            else:
                response = ''
            tgt = self.resp2str(response)

            if len(tgt) > 0 and tgt[-1] in '?？':
                self.anom_stat['n_qst_type_resp'] += 1
                if self.skip_ask:
                    # Skip question-type response
                    continue

            itg_sample = {
                'src_pre': src_pre,
                'src_post': src_post,
                'src_img': src_img,
                'tgt': tgt,
            }
            if (not self.anom and itg_sample['tgt'] != '') or self.run_type == 'infer':
                itg_samples.append(itg_sample)

        print('[*] Anomaly stat: his=%d/%d, know=%d/%d, user=%d/%d, resp=%d/%d, qst_type_resp=%d, anom_img=%d' %
              (self.anom_stat['n_long_his'], self.anom_stat['n_tot_his'],
               self.anom_stat['n_long_know'], self.anom_stat['n_tot_know'],
               self.anom_stat['n_long_user'], self.anom_stat['n_tot_user'],
               self.anom_stat['n_long_resp'], self.anom_stat['n_tot_resp'],
               self.anom_stat['n_qst_type_resp'], self.anom_stat['n_anom_img']))

        print('[+] Convert from dataset:', len(data), 'to samples:', len(itg_samples))
        return itg_samples

    def user2str(self, user_utt):
        user_utt_str = str(user_utt)
        user_utt_str = remove_whilespace(user_utt_str)
        user_utt_str = user_utt_str.rstrip(' ?？')
        if not user_utt_str.startswith(self.user_token):
            user_utt_str = self.user_token + user_utt_str

        if len(user_utt_str) > self.max_user_len:
            self.anom_stat['n_long_user'] += 1
            user_utt_str = user_utt_str[:self.max_user_len]
            self.anom = True
        self.anom_stat['n_tot_user'] += 1
        return user_utt_str

    def post2str(self, post):
        post_str = str(post)
        post_str = remove_whilespace(post_str)
        if not post_str.endswith(self.resp_token):
            post_str = post_str + self.resp_token
        assert len(post_str) < self.max_post_len
        return post_str

    def resp2str(self, resp_utt):
        resp_utt_str = str(resp_utt)
        if len(resp_utt_str) > self.max_resp_len:
            self.anom_stat['n_long_resp'] += 1
            resp_utt_str = resp_utt_str[:self.max_resp_len]
            self.anom = True
        self.anom_stat['n_tot_resp'] += 1
        return resp_utt_str

    def know2str(self, know):
        know_str = str(know)
        if not know_str.startswith(self.know_token):
            know_str = self.know_token + know_str

        if len(know_str) > self.max_know_len:
            self.anom_stat['n_long_know'] += 1
            know_str = know_str[:self.max_know_len]
            # self.anom = True
        self.anom_stat['n_tot_know'] += 1
        return know_str

    def his2str(self, his_utt):
        his_utt_str = str(his_utt)
        his_utt_str = remove_whilespace(his_utt_str)

        if len(his_utt_str) > self.max_his_len:
            self.anom_stat['n_long_his'] += 1
            # self.anom = True
            new_his_utt_str = his_utt_str[-self.max_his_len:]
            mat = re.search('^.*?(<用户>|<系统>)(.*)$', new_his_utt_str)
            if mat is not None:
                his_utt_str = mat.group(1) + mat.group(2)
            else:
                his_utt_str = '<系统>' + his_utt_str.split('<系统>')[-1]
                his_utt_str = his_utt_str[:self.max_his_len]
        self.anom_stat['n_tot_his'] += 1

        if not his_utt_str.startswith(self.his_token):
            his_utt_str = self.his_token + his_utt_str
        return his_utt_str

    def post_proc(self, data):
        for out in data:
            out['pred'] = self.get_tgt(out['pred'])
            del out['src']['src_img']
        return data

    def get_tgt(self, text):
        tgt_prompt = self.resp_token
        res_pieces = text.split(tgt_prompt)
        if len(res_pieces) >= 2:
            res_txt = res_pieces[1]
        else:
            res_txt = text
        return res_txt
