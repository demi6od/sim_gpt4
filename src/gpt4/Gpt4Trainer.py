import torch
import logging
import os
from tqdm import tqdm

from src.gpt4.Gpt4Model import Gpt4Model

from transformers import get_linear_schedule_with_warmup

from src.utils.Dataset import Dataset
from src.utils.utils import print_params
from src.utils.Metric import Metric
from transformers import Blip2Processor, Blip2Model

logger = logging.getLogger('Gpt4Trainer')

class Gpt4Trainer:
    def __init__(self, args):
        self.args = args

        # For debug
        # torch.set_printoptions(threshold=10_000)

        self.max_score = 0
        self.min_loss = float('inf')
        self.min_loss_score = 0
        self.max_epoch = 0
        self.min_epoch = 0
        self.max_batch = 0
        self.min_batch = 0
        self.score_his = []
        self.loss_his = []

        self.metric = Metric()

        self.blip2_processor = Blip2Processor.from_pretrained(args.blip2_model_name)

        self.gpt4_model = Gpt4Model(args)
        print_params(self.gpt4_model)

        logger.info(f'[+] Image num_tokens {self.gpt4_model.blip2_model.config.num_query_tokens}')

        if torch.cuda.is_available():
            self._model_device = 'cuda'
        else:
            self._model_device = 'cpu'
            logger.warning('No CUDA found!')

        if args.load_model:
            self._load_model()

        if self._model_device == 'cuda' and args.model_type != 'bloomz':
            self._move_model_to_cuda(args.device_id)

        self.tokenizer = self.gpt4_model.get_tokenizer()
        self.dataset = Dataset(args, self.tokenizer)
        self.data_loader = self.dataset.get_data_loader()

    def _save_model(self):
        self.gpt4_model.save_model()

    def _load_model(self):
        self.gpt4_model.load_model()

    def _move_model_to_cuda(self, device_id):
        logger.info('Moving gpt4 model model to CUDA [%d]' % device_id)
        torch.cuda.set_device(device_id)
        self.gpt4_model.to(self._model_device)

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.gpt4_model.parameters(), lr=self.args.model_lr)

        t_total = len(self.data_loader['train']) * self.args.num_epoch
        logger.info('[+] Total step = %.1f k' % (t_total / 1e3))

        warmup_proportion = 0.1
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(t_total * warmup_proportion), t_total)

    def run_train(self):
        self.set_optimizer()

        for epoch in tqdm(range(self.args.num_epoch)):
            self.train(epoch)
            self.evaluate(epoch, batch_id=-1)

    def run_test(self):
        out = self.validate('test')
        logger.info('[+] Test score=%.3f' % out['score'])

    def run_infer(self, inputs):
        if torch.cuda.is_available() and self.args.model_type != 'bloomz':
            torch.cuda.set_device(self.args.device_id)
        inputs = self.dataset.read_data(inputs, 'infer')
        out = self.validate('infer')

        pred_dics = []
        for inp, pred in zip(out['input_lst'], out['pred_lst']):
            pred_dic = {
                'src': inp,
                'pred': pred
            }
            pred_dics.append(pred_dic)
        return pred_dics

    def train(self, epoch_id):
        logger.info('Train batch num = %d' % len(self.data_loader['train']))
        for batch_id, data in enumerate(tqdm(self.data_loader['train'])):
            self.gpt4_model.train()

            inputs = self._get_input_id(data)
            if inputs is None:
                continue

            out = self.gpt4_model(inputs)
            loss = out['loss']

            loss = loss.mean()
            loss.backward()

            if self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.gpt4_model.parameters(), max_norm=5.0)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if batch_id > 0 and batch_id % self.args.print_step == 0:
                logger.info('epoch=%d, batch=%d, loss=%.5f' % (epoch_id, batch_id, loss.item()))
                self.evaluate(epoch_id, batch_id)

    @torch.no_grad()
    def evaluate(self, epoch, batch_id):
        out = self.validate('dev')
        loss = out['loss']
        loss = loss.mean()
        self.loss_his.append(round(loss.item(), 5))
        score = out['score']
        self.score_his.append(score)

        logger.info('[+] Epoch = %d, Batch = %d' % (epoch, batch_id))
        logger.info('[+] Score history: %s' % self.score_his)
        logger.info('[+] Best history eval score=%.3f at epoch=%d, batch=%d' %
                    (self.max_score, self.max_epoch, self.max_batch))
        if score > self.max_score:
            logger.info('[*] New best score!')
            self.max_score = score
            self.max_epoch = epoch
            self.max_batch = batch_id

        logger.info('[+] Eval loss history: %s' % self.loss_his)
        logger.info('[+] Minimum history eval loss=%.3f score=%.3f at epoch=%d, batch=%d' %
                    (self.min_loss, self.min_loss_score, self.min_epoch, self.min_batch))
        if loss < self.min_loss:
            logger.info('[*] New minimum loss!')
            self.min_loss = loss
            self.min_loss_score = score
            self.min_epoch = epoch
            self.min_batch = batch_id
            self._save_model()

    @torch.no_grad()
    def validate(self, data_t):
        pred_txt_lst = []
        gold_txt_lst = []
        input_txt_lst = []
        loss_lst = []
        self.gpt4_model.eval()
        logger.info('Validate batch num = %d' % len(self.data_loader[data_t]))
        for batch_id, data in enumerate(tqdm(self.data_loader[data_t])):
            inputs = self._get_input_id(data)
            gold_txts = [item['tgt'] for item in data]
            if self.args.run_type == 'infer':
                input_txt_lst.extend(data)

            out = self.gpt4_model(inputs)

            pred = self._get_pred_label(out['gen_ids'], gold_txts)
            pred_txt_lst += pred['texts']
            gold_txt_lst += gold_txts
            loss_lst.append(out['loss'])

        out = self._cal_score(input_txt_lst, pred_txt_lst, gold_txt_lst, loss_lst)
        return out

    def _get_input_id(self, samples):
        if self.args.model_type in ['t5', 'mt0']:
            inputs = self._get_input_id_encdec(samples)
        elif self.args.model_type in ['gpt', 'bloomz']:
            inputs = self._get_input_id_dec(samples)
        return inputs

    def _get_input_id_encdec(self, samples):
        srcs = []
        tgts = []
        for sample in samples:
            srcs.append(sample['src'] + self.gpt4_model.eos_token)
            tgts.append(sample['tgt'] + self.gpt4_model.eos_token)

        src_ids = self.tokenizer(
            text=srcs,
            padding='max_length',
            max_length=self.args.max_src_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        src_tensors = src_ids.to(self._model_device)

        tgt_ids = self.tokenizer(
            text=tgts,
            padding='max_length',
            max_length=self.args.max_tgt_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        tgt_inp_ids = tgt_ids['input_ids']
        tgt_inp_tensors = tgt_inp_ids.to(self._model_device)

        inputs = {
            'src': src_tensors,
            'tgt_inp_ids': tgt_inp_tensors
        }
        return inputs

    def _get_input_id_dec(self, samples):
        srcs = []
        src_imgs = []
        tgts = []
        src_img = self.gpt4_model.bos_token * self.gpt4_model.blip2_model.config.num_query_tokens
        for sample in samples:
            if self.args.run_type == 'train':
                srcs.append(sample['src_pre'] + src_img + sample['src_post'] + sample['tgt'] + self.gpt4_model.eos_token)
            else:
                srcs.append(sample['src_pre'] + src_img + sample['src_post'])  # Generate
            src_imgs.append(sample['src_img'])
            tgts.append(sample['tgt'] + self.gpt4_model.eos_token)

        src_ids = self.tokenizer(
            text=srcs,
            padding='max_length',
            max_length=self.args.max_src_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        src_tensors = src_ids.to(self._model_device)

        tgt_ids = self.tokenizer(
            text=tgts,
            padding='max_length',
            max_length=self.args.max_tgt_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        tgt_inp_ids = tgt_ids['input_ids']
        tgt_inp_tensors = tgt_inp_ids.to(self._model_device)

        try:
            src_img_tensors = self.blip2_processor(images=src_imgs, return_tensors="pt").to(self._model_device)
            # src_img_tensors = self.blip2_processor(images=src_imgs, return_tensors="pt").to(self._model_device, torch.float16)
        except Exception as e:
            logger.warning(f'[*] blip2_processor error: {e}')
            return None

        inputs = {
            'src': src_tensors,
            'src_imgs': src_img_tensors,
            'tgt_inp_ids': tgt_inp_tensors
        }
        return inputs

    def _cal_score(self, input_txt_lst, pred_txt_lst, gold_txt_lst, loss_lst):
        if self.args.run_type == 'infer':
            score = None
        else:
            if self.args.metric == 'bleu':
                score = self.metric.bleu_score(self.tokenizer, pred_txt_lst, gold_txt_lst)
            elif self.args.metric == 'rouge':
                score = self.metric.rouge_score(self.tokenizer, pred_txt_lst, gold_txt_lst)
            elif self.args.metric == 'f1':
                score = self.metric.cal_f1(pred_txt_lst, gold_txt_lst)

        if loss_lst[0] is None:
            loss = None
        else:
            loss = sum(loss_lst) / (len(loss_lst) + 1e-5)
        out = {
            'input_lst': input_txt_lst,
            'pred_lst': pred_txt_lst,
            'gold_lst': gold_txt_lst,
            'loss': loss,
            'score': score,
        }
        return out

    def _get_pred_label(self, gen_ids, gold_texts):
        ''' Get predicted texts from gen_ids '''

        texts = []
        assert len(gen_ids) == len(gold_texts)
        for ids, gold_text in zip(gen_ids, gold_texts):
            if self.args.model_type in ['gpt', 'bloomz']:
                if self.args.run_type == 'train':
                    tgt_ids = self.tokenizer.encode(gold_text)
                    tgt_len = len(tgt_ids) + 1
                    ids = ids[-tgt_len:]
                else:
                    ids = ids
            text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            text = text.replace(' ', '')
            texts.append(text)

        pred = {
            'texts': texts,
        }
        return pred
