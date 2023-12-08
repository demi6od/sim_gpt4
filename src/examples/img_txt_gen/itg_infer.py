import os
import sys
from pprint import pprint

import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '../../..')
sys.path.append(PROJECT_DIR)

from src.gpt4.Gpt4Infer import Gpt4Infer
from src.utils.utils import write_file

def main():
    sys.argv = ['', '../../params/img_txt_gen/itg_bloomz_infer.json']
    args = {
        'project_dir': PROJECT_DIR,
    }

    nlg_infer = Gpt4Infer(args)

    samples = [
        {
            "id": "COCO_train2014_000000174188",
            "caption": "酒店大厅中央有一排喷泉，上面挂着各种颜色的伞。",
            'dialog_his': '',
            'knows': [""],
            'query': '图片里画了什么？',
            "dataset": "coco"
        },
        {
            "id": "COCO_val2014_000000280688",
            "caption": "在桌子上的键盘旁边的咖啡杯。",
            'dialog_his': '',
            'knows': [""],
            'query': '能描述一下图片的内容吗？',
            "dataset": "coco"
        },
        {
            "id": "COCO_train2014_000000297058",
            "caption": "建筑物下摆放着一个指路标志",
            'dialog_his': '',
            'knows': [""],
            'query': '请详细描述一下图片内容',
            "dataset": "coco"
        },
        {
            "id": "COCO_train2014_000000063252",
            "caption": "一只狗坐在家门口的台阶上。",
            'dialog_his': '',
            'knows': [""],
            'query': '请解释一下为什么这幅图片比较搞笑',
            "dataset": "coco"
        },
        {
            "id": "COCO_val2014_000000379014",
            "caption": "夜晚的街道上有信号灯，标志牌等等",
            'dialog_his': '',
            'knows': [""],
            'query': '能写一首关于这幅图片的诗歌吗？',
            "dataset": "coco"
        },
        {
            "id": "COCO_val2014_000000530934",
            "caption": "在雪地中有一个戴着帽子和墨镜的男人，口里含着牙刷。",
            'dialog_his': '',
            'knows': [""],
            'query': '请为这幅图写一个广告文案',
            "dataset": "coco"
        },
        {
            "id": "COCO_val2014_000000525162",
            "caption": "一个人乘着皮艇在水中滑行。",
            'dialog_his': '',
            'knows': [""],
            'query': '请根据这幅图写一篇散文',
            "dataset": "coco"
        },
        {
            "id": "COCO_train2014_000000077018",
            "caption": "一些人在海滩上玩耍、放风筝。",
            'dialog_his': '',
            'knows': [""],
            'query': '请根据这幅图写一篇小说',
            "dataset": "coco"
        },
    ]
    outs = nlg_infer.batch_run(samples)

    assert len(outs) == len(samples)
    results = []
    for sample, out in zip(samples, outs):
        res = {
            'caption': sample['caption'],
            'query': sample['query'],
            'img_id': sample['id'],
            'response': out['pred']
        }
        results.append(res)

    write_file(nlg_infer.args.output_dir, 'infer_out.json', results)
    df = pd.DataFrame(results)
    df.to_excel(os.path.join(nlg_infer.args.output_dir, 'infer_out.xlsx'))

    pprint(results)
    print('[+] Finish')


if __name__ == '__main__':
    main()
