import logging

from src.process.img_txt_gen.Processor import Processor as Itg_processor
from src.utils.utils import read_args, parse_args, log_config
from src.gpt4.Gpt4Trainer import Gpt4Trainer

logger = logging.getLogger('Gpt4Infer')

class Gpt4Infer:
    def __init__(self, infer_args):
        infer_args['run_type'] = 'infer'
        self.args = read_args(infer_args)
        self.args = parse_args(self.args)

        log_config(self.args, 'infer')

        self.nlg_trainer = Gpt4Trainer(args=self.args)

        proc_dic = {
            'img_txt_gen': Itg_processor,
        }
        self.processor = proc_dic[self.args.task](run_type='infer', skip_ask=False, args=self.args)

    def run(self, ins):
        outs = self.batch_run([ins])
        return outs[0]

    def batch_run(self, ins):
        inputs = self.processor.pre_proc(ins)
        outputs = self.nlg_trainer.run_infer(inputs)
        outs = self.processor.post_proc(outputs)
        return outs
