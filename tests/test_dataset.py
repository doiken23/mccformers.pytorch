#!/usr/bin/env python3

import torch

from datasets.cc_dataset import RCCDataset
from datasets.original_cmc_dataset import CaptionDataset


def test_rcc_dataset():
    dataset = RCCDataset("./data/clevr", split="train", batch_size=128, seq_per_img=1)

    (
        d_feature,
        n_feature,
        q_feature,
        seq,
        neg_seq,
        mask,
        neg_mask,
        aux_label_pos,
        aux_label_neg,
        d_img_path,
        n_img_path,
        q_img_path,
    ) = dataset[0]

    assert isinstance(d_feature, torch.Tensor)
    assert isinstance(n_feature, torch.Tensor)
    assert isinstance(q_feature, torch.Tensor)

    assert isinstance(seq, torch.Tensor)
    assert isinstance(neg_seq, torch.Tensor)

    assert isinstance(d_img_path, str)
    assert isinstance(n_img_path, str)
    assert isinstance(q_img_path, str)

    for i, data in enumerate(dataset):
        seq = data[3]
        neg_seq = data[4]

        assert seq[0, seq.argmin(1) - 1] == 3
        assert neg_seq[0, neg_seq.argmin(1) - 1] == 3

        if i == 99:
            break


def test_cmc_dataset():
    data_folder = "data/original_clevr_multi"
    data_name = "3dcc_5_cap_per_img_0_min_word_freq"
    captions_per_image = 5
    dataset_name = "MOSCC"
    for split in ["TRAIN", "TEST"]:
        dataset = CaptionDataset(data_folder, data_name, split, captions_per_image, dataset_name)

        for i, data in enumerate(dataset):
            if split == "TRAIN":
                img1, img2, caption, caplen = data
            else:
                img1, img2, caption, caplen, all_captions = data

            assert isinstance(img1, torch.Tensor)
            assert img1.size() == (1024, 14, 14)

            assert isinstance(img2, torch.Tensor)
            assert img2.size() == (1024, 14, 14)

            assert isinstance(caption, torch.Tensor)
            assert isinstance(caplen, torch.Tensor)

            if split == "TEST":
                assert isinstance(all_captions, torch.Tensor)
                assert len(all_captions) == captions_per_image

            if i == 99:
                break
