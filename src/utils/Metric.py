import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger('Metric')

class Metric:
    def __init__(self):
        self.rouge = Rouge()

    @staticmethod
    def cal_f1(preds, golds, log=True):
        true_pos = 0
        false_pos = 0
        false_neg = 0
        epson = 1e-5
        for pred, gold in zip(preds, golds):
            if pred == '1' and gold == '1':
                true_pos += 1
            elif pred != '0' and gold == '0':
                false_pos += 1
            elif pred != '1' and gold == '1':
                false_neg += 1

        precision = 1 if (true_pos + false_pos) == 0 else true_pos / (true_pos + false_pos)
        recall = 1 if (true_pos + false_neg) == 0 else true_pos / (true_pos + false_neg)
        if log:
            logger.info('Precision = %.3f' % precision)
            logger.info('Recall = %.3f' % recall)
        # f1 = 1 / (0.5 * (1 / precision + 1 / recall))
        f1 = 2 * precision * recall / (precision + recall + epson)
        f1 = round(f1, 3)
        return f1

    def bleu_score(self, tokenizer, preds, golds):
        smoothie = SmoothingFunction().method4
        score = 0
        for idx in range(len(preds)):
            ref = tokenizer.tokenize(golds[idx])
            hyp = tokenizer.tokenize(preds[idx])
            try:
                score += sentence_bleu([ref], hyp, smoothing_function=smoothie)
                # score += sentence_bleu([ref], hyp)
            except Exception as e:
                logger.error(e)
        avg_score = 1 if len(preds) == 0 else score / len(preds)
        avg_score = round(avg_score, 3)
        return avg_score

    def cal_sent_acc(self, pred, gold):
        cor = 0
        tot = 0
        for char in pred:
            if char in gold:
                cor += 1
            tot += 1
        acc = 1 if tot == 0 else cor / tot
        return acc

    def cal_sent_f1(self, pred, gold):
        precision = self.cal_sent_acc(pred, gold)
        recall = self.cal_sent_acc(gold, pred)

        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        f1 = round(f1, 3)
        return f1

    def f1_score(self, preds, golds):
        tot_f1 = 0
        assert len(preds) == len(golds)
        for pred, gold in zip(preds, golds):
            tot_f1 += self.cal_sent_f1(pred, gold)
        avg_f1 = 1 if len(preds) == 0 else round(tot_f1 / len(preds), 3)
        return avg_f1

    def cal_rouge(self, tokenizer, pred, gold):
        ref = ' '.join(tokenizer.tokenize(gold))
        hyp = ' '.join(tokenizer.tokenize(pred))
        try:
            r_score = self.rouge.get_scores(hyp, ref, avg=True)
            score = r_score['rouge-l']['f']
        except Exception as e:
            score = 0
            logger.error(e)
        return score

    def cal_rouge_raw(self, pred, gold):
        ref = ' '.join(list(gold))
        hyp = ' '.join(list(pred))
        try:
            r_score = self.rouge.get_scores(hyp, ref, avg=True)
            score = r_score['rouge-l']['f']
        except Exception as e:
            score = 0
            logger.error(e)
        return score

    def rouge_score(self, tokenizer, preds, golds):
        score = 0
        for idx in range(len(preds)):
            score += self.cal_rouge(tokenizer, preds[idx], golds[idx])
        avg_score = 1 if len(preds) == 0 else score / len(preds)
        avg_score = round(avg_score, 3)
        return avg_score
