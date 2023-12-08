import os
import json
import logging
import random
from src.process.img_txt_gen.Processor import Processor as Itg_processor

logger = logging.getLogger('Dataset')

class Dataset:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.batch_size = {
            'train': args.train_batch_size,
            'dev': args.dev_batch_size,
            'test': args.test_batch_size,
            'infer': args.infer_batch_size,
        }
        self.max_sample = {
            'train': args.max_train,
            'dev': args.max_dev,
            'test': args.max_test,
            'infer': args.max_infer,
        }

        if args.run_type == 'train' or args.run_type == 'finetune':
            self.data_types = ['train', 'dev']
        elif args.run_type == 'test':
            self.data_types = ['test']
        elif args.run_type == 'infer':
            self.data_types = ['infer']
        else:
            logger.error('[-] else not catch!')

        self.processor = Itg_processor(run_type=self.args.run_type, skip_ask=False, args=args)

        self.data_dic = {}
        self.samples_dic = {}
        self.data_loader = {}

        if self.args.run_type != 'infer':
            for data_t in self.data_types:
                data_file = os.path.join(args.data_dir, data_t + '.json')
                logger.info('Read data file from %s...' % data_file)
                data = self._read(data_file)
                self.data_dic[data_t] = data[:int(self.max_sample[data_t])]

                self.generate_samples(data_t, shuffle=True)

    def generate_samples(self, data_t, shuffle):
        if shuffle:
            random.shuffle(self.data_dic[data_t])

        self.samples_dic[data_t] = self.processor.pre_proc(self.data_dic[data_t])
        self.samples_dic[data_t] = self.samples_dic[data_t][:int(self.max_sample[data_t])]
        self.data_loader[data_t] = self.batchify(self.samples_dic[data_t], self.batch_size[data_t])
        return self.data_loader[data_t]

    def get_data(self):
        return self.samples_dic

    def get_data_loader(self):
        return self.data_loader

    def read_data(self, data, data_t):
        self.samples_dic[data_t] = data[:int(self.max_sample[data_t])]
        self.data_loader[data_t] = self.batchify(self.samples_dic[data_t], self.batch_size[data_t])
        return self.samples_dic[data_t]

    def _read(self, json_file):
        with open(json_file, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        return data

    def batchify(self, samples, batch_size, shuffle=False):
        """ Batchify samples with a batch size """

        if shuffle:
            random.shuffle(samples)

        batch_lst = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:(i + batch_size)]
            batch_lst.append(batch)

        assert(sum([len(batch) for batch in batch_lst]) == len(samples))
        return batch_lst
