from PIL import Image
import requests
import argparse
import os
import sys
import random
import json
import logging
import time
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from rouge import Rouge
# from LAC import LAC

# # 装载词语重要性模型
# lac = LAC(mode='rank')
#
# # 单个样本输入，输入为Unicode编码的字符串
# text = u"LAC是个优秀的分词工具"
# rank_result = lac.run(text)
# print(rank_result)
#
# # 批量样本输入, 输入为多个句子组成的list，平均速率会更快
# texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
# rank_result = lac.run(texts)
# print(rank_result)



# model_out = ["百度是一家高科技公司"]
#
# reference = ["百度是公司"]
# # reference = ["he began his premiership by forming a five-man war cabinet which included chamberlain as lord president of the council",
# #              "the siege of lilybaeum lasted from 250 to 241 bc, as the roman army laid siege to the carthaginian-held sicilian city of lilybaeum",
# #              "the original mission was for research into the uses of deep ocean water in ocean thermal energy conversion (otec) renewable energy production and in aquaculture"]
# rouge = Rouge()
# score = rouge.get_scores(model_out, reference, avg=True)
# f1 = score['rouge-l']['f']
# print(score)
# print(f1)
# def chunkstring(string, length):
#     return [string[(0 + i):(length + i)] for i in range(0, len(string), length)]
#
# b = chunkstring('abcdefg', 30)
# print(b)
#
# import re
# chatgpt_reg = re.compile(r'\bgpt\b', re.IGNORECASE)
# text = chatgpt_reg.sub('SenseLM', 'it is gpt model')
# print(text)
#
# a = 'aabbaabbaabb'
# b = a.split('bb', 1)
# print(b)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
a = torch.tensor([0.0123456], dtype=torch.float16)
print(a)
a = a.type(torch.float32)
print(a)

print('[+] finish')