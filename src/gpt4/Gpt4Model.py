import logging
import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import Blip2Processor, Blip2Model

logger = logging.getLogger('Gpt4Model')

class Gpt4Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        if torch.cuda.is_available():
            self._model_device = 'cuda'
        else:
            self._model_device = 'cpu'
            logger.warning('No CUDA found!')

        logger.info('Loading backbone lang model from %s' % args.lang_model_name)
        if self.args.model_type in ['mt0', 't5']:
            if self.args.model_type == 'mt0' and self.args.run_type != 'train':
                self.lang_model = AutoModelForSeq2SeqLM.from_pretrained(args.lang_model_name, torch_dtype='auto')
            else:
                self.lang_model = AutoModelForSeq2SeqLM.from_pretrained(args.lang_model_name)
        elif self.args.model_type in ['gpt', 'bloomz']:
            if self.args.model_type == 'bloomz' and not self.args.lit:
                if self.args.run_type == 'train':
                    self.lang_model = AutoModelForCausalLM.from_pretrained(args.lang_model_name, device_map='auto')
                else:
                    self.lang_model = AutoModelForCausalLM.from_pretrained(args.lang_model_name, device_map='auto',
                                                                      torch_dtype='auto')
            else:
                self.lang_model = AutoModelForCausalLM.from_pretrained(args.lang_model_name)

        if not self.args.use_raw_lm:
            self.load_nlg_base()

        for name, param in self.lang_model.named_parameters():
            param.requires_grad = False

        logger.info('Loading backbone blip2 model from %s' % args.blip2_model_name)
        # self.blip2_model = Blip2Model.from_pretrained(args.blip2_model_name, torch_dtype=torch.float16).to('cuda')
        self.blip2_model = Blip2Model.from_pretrained(args.blip2_model_name).to(self._model_device)
        for name, param in self.blip2_model.named_parameters():
            param.requires_grad = False

        self.lang_projection = nn.Linear(self.blip2_model.config.qformer_config.hidden_size,
                                         self.lang_model.config.hidden_size).to(self._model_device)

        logger.info('Loading tokenizer from %s' % args.lang_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.lang_model_name)
        if self.args.model_type in ['gpt', 'bloomz']:
            self.tokenizer.padding_side = 'left'

        if self.args.model_type in ['t5', 'gpt']:
            self.eos_token = self.tokenizer.sep_token
            self.eos_token_id = self.tokenizer.sep_token_id
        elif self.args.model_type in ['mt0', 'bloomz']:
            self.eos_token = self.tokenizer.eos_token
            self.eos_token_id = self.tokenizer.eos_token_id
            self.bos_token = self.tokenizer.bos_token
            self.bos_token_id = self.tokenizer.bos_token_id

        if args.parallel:
            device_id_end = min(args.device_id + args.device_cnt, torch.cuda.device_count()) - 1
            device_ids = range(args.device_id, device_id_end + 1)
            self.lang_model = nn.DataParallel(self.lang_model, device_ids=device_ids)
            logger.info('# GPU cnt = %d' % len(device_ids))
            logger.info('Parallel model on devices: [%d] - [%d]' % (args.device_id, device_id_end))

        self._init_weights()

    def save_model(self):
        model_name = 'gpt4_model_' + self.args.model_type + '.pt_' + self.args.now
        path = os.path.join(self.args.output_dir, model_name)
        logger.info('Saving model to %s...' % path)
        torch.save(self.lang_projection.state_dict(), path)

    def load_nlg_base(self):
        path_nlg = os.path.join(self.args.model_dir, 'nlg_model_' + self.args.model_type + '_base.pt')
        logger.info('Loading nlg model from %s...' % path_nlg)
        if self._model_device == 'cpu':
            self.lang_model.load_state_dict(torch.load(path_nlg, map_location=torch.device('cpu')), strict=True)
        elif self._model_device == 'cuda':
            self.lang_model.load_state_dict(
                torch.load(path_nlg, map_location=torch.device('cuda:' + str(self.args.device_id))),
                strict=True)

    def load_model(self):
        path_proj = os.path.join(self.args.model_dir, 'gpt4_model_' + self.args.model_type + '.pt')
        logger.info('Loading gpt4 model from %s...' % path_proj)
        if self._model_device == 'cpu':
            self.lang_projection.load_state_dict(torch.load(path_proj, map_location=torch.device('cpu')), strict=True)
        elif self._model_device == 'cuda':
            self.lang_projection.load_state_dict(
                torch.load(path_proj, map_location=torch.device('cuda:' + str(self.args.device_id))),
                strict=True)

    def get_tokenizer(self):
        return self.tokenizer

    def _init_weights(self):
        names = []
        for name, param in self.named_parameters():
            if param.requires_grad and 'lang_projection' in name:
                names.append(name)
                torch.nn.init.normal_(param.data, mean=0, std=0.01)
        logger.info('Initial params: {}'.format(names))

    def forward(self, inputs):
        with torch.no_grad():
            img_inputs = inputs['src_imgs']
            qformer_outputs = self.blip2_model.get_qformer_features(**img_inputs)
            query_output = qformer_outputs[0]

        # use the lang model, conditioned on the query outputs and the prompt
        # query_output = query_output.to(torch.float32)
        lang_model_inputs = self.lang_projection(query_output)

        input_ids = inputs['src']['input_ids']
        inputs_embed_tots = []
        for idx, input_id in enumerate(input_ids):
            img_idx = (input_id == self.bos_token_id).nonzero(as_tuple=True)[0]
            img_start_idx = img_idx[0]
            img_end_idx = img_idx[-1]

            input_pre = input_id[:img_start_idx]
            inputs_embed_pre = self.lang_model.get_input_embeddings()(input_pre)
            input_post = input_id[(img_end_idx + 1):]
            inputs_embed_post = self.lang_model.get_input_embeddings()(input_post)

            inputs_embed_img = lang_model_inputs[idx].to(inputs_embed_pre.device)

            inputs_embed = torch.cat([inputs_embed_pre, inputs_embed_img, inputs_embed_post], dim=0)
            inputs_embed_tots.append(inputs_embed)

        inputs_embed_tots = torch.stack(inputs_embed_tots)
        attention_mask = inputs['src']['attention_mask']
        assert inputs_embed_tots.shape[1] == attention_mask.shape[1]

        if self.args.run_type == 'train':
            labels = inputs['tgt_inp_ids'].clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.lang_model(
                inputs_embeds=inputs_embed_tots,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs['loss']
            logits = outputs['logits']
            # logits = [batch_size, seq_len, vocab_size]

            max_res = torch.max(logits, dim=-1, keepdim=False)
            gen_ids = max_res[1]
        elif self.args.run_type in ['test', 'infer']:
            gen_ids = self.lang_model.generate(
                inputs_embeds=inputs_embed_tots,
                attention_mask=attention_mask,
                max_length=self.args.max_tgt_len,
                eos_token_id=self.eos_token_id,
                do_sample=self.args.do_sample,
                top_p=self.args.top_p,  # sample from [words] prob cover top_p
                no_repeat_ngram_size=5,
            )
            loss = None
        out = {
            'loss': loss,
            'gen_ids': gen_ids,
        }
        return out
