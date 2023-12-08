import os
import sys
import logging

# Set PROJECT_DIR as package search path

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_DIR)

from utils.utils import read_args, parse_args, log_config, setseed
from gpt4.Gpt4Trainer import Gpt4Trainer

logger = logging.getLogger('run_nlg')

def main():
    args = read_args()
    args = parse_args(args)

    setseed(args.seed)

    if args.run_type == 'train':
        log_config(args, 'train')
    else:
        log_config(args, 'eval')

    logger.info(sys.argv)
    logger.info(args)

    nlg_trainer = Gpt4Trainer(args)
    if args.run_type == 'train' or args.run_type == 'finetune':
        nlg_trainer.run_train()
    else:
        nlg_trainer.run_test()

    logger.info('[+] End!')


if __name__ == '__main__':
    main()
