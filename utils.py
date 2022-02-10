import argparse
import logging
import os
from datetime import datetime
from logging import DEBUG, INFO, NOTSET, FileHandler, StreamHandler
from typing import Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


def load_configs() -> DictConfig:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("-o", "--overrides", nargs="*", help="override options")
    args = parser.parse_args()

    # load config
    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(cfg, cli)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    # init_distributed_mode(cfg)

    return cfg


def create_logger(output_dir: Union[str, bytes, os.PathLike]):
    # stream handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)

    # file handler
    file_handler = FileHandler(
        str(output_dir.joinpath("log_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"))))
    )
    file_handler.setLevel(DEBUG)

    # root logger
    logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])


def compute_accuracy(pred: Tensor, gt: Tensor, ignore: int = 0):
    """
    pred (torch.Tensor): predicted words shape of [L, N]
    gt (torch.Tensor): GT words shape of [L, N]
    ignore (int): ignored label
    """
    mask = gt != ignore
    tp = torch.logical_and(pred == gt, mask)

    return tp.sum() / mask.sum()


def decode_seq(seq, idx_to_word, start_idx=2, end_idx=3, pad_idx=None):
    words = []
    for s in seq:
        if s == start_idx:  # <START>
            continue
        if s == end_idx:  # <END>
            break
        if pad_idx is not None and s == pad_idx:  # <PAD>
            continue
        words.append(idx_to_word[s])
    return " ".join(words)
