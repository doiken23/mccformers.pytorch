#!/usr/bin/env python3

import torch

from datasets import RCCDataset


def test_rcc_dataset():
    dataset = RCCDataset('./data', split='train', batch_size=128, seq_per_img=1)

    d_feature, n_feature, q_feature, \
        seq, neg_seq, mask, neg_mask, aux_label_pos, aux_label_neg, \
        d_img_path, n_img_path, q_img_path = dataset[0]

    assert isinstance(d_feature, torch.Tensor)
    assert isinstance(n_feature, torch.Tensor)
    assert isinstance(q_feature, torch.Tensor)

    assert isinstance(seq, torch.Tensor)
    assert isinstance(neg_seq, torch.Tensor)

    assert isinstance(d_img_path, str)
    assert isinstance(n_img_path, str)
    assert isinstance(q_img_path, str)

    for data in dataset:
        seq = data[3]
        neg_seq = data[4]

        assert seq[0, seq.argmin(1) - 1] == 3
        assert neg_seq[0, neg_seq.argmin(1) - 1] == 3
