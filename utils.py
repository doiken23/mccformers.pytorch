import argparse

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


def compute_accuracy(pred: Tensor, gt: Tensor, ignore: int = 0):
    """
    pred (torch.Tensor): predicted words shape of [L, N]
    gt (torch.Tensor): GT words shape of [L, N]
    ignore (int): ignored label
    """
    mask = gt != ignore
    tp = torch.logical_and(pred == gt, mask)

    return tp.sum() / mask.sum()


def decode_seq(seq, idx_to_word):
    words = []
    for s in seq:
        if s == 2:  # <START>
            continue
        if s == 3:  # <END>
            break
        words.append(idx_to_word[s])
    return " ".join(words)
