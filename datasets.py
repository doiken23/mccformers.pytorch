import json
import random
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


class RCCDataset(Dataset):

    shapes = set(["ball", "block", "cube", "cylinder", "sphere"])
    sphere = set(["ball", "sphere"])
    cube = set(["block", "cube"])
    cylinder = set(["cylinder"])

    colors = set(["red", "cyan", "brown", "blue", "purple", "green", "gray", "yellow"])

    materials = set(["metallic", "matte", "rubber", "shiny", "metal"])
    rubber = set(["matte", "rubber"])
    metal = set(["metal", "metallic", "shiny"])

    type_to_label = {
        "color": 0,
        "material": 1,
        "add": 2,
        "drop": 3,
        "move": 4,
        "no_change": 5,
    }

    def __init__(self, data: str, split: str, batch_size: int = 128, seq_per_img: int = 1):
        self.data = Path(data)

        print("Speaker Dataset loading vocab json file: ", "data/vocab.json")
        self.vocab_json = self.data.joinpath("vocab.json")
        with self.vocab_json.open("r") as f:
            self.word_to_idx = json.load(f)
        self.idx_to_word = {}
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
        self.vocab_size = len(self.idx_to_word)
        print("vocab size is ", self.vocab_size)

        with self.data.joinpath("type_mapping.json").open("r") as f:
            self.type_mapping = json.load(f)
        self.type_to_img = {}
        for k, v in self.type_mapping.items():
            self.type_to_img[k] = set([int(x.split(".")[0]) for x in v])

        # feature paths
        self.d_feat_dir = self.data.joinpath("features")
        self.s_feat_dir = self.data.joinpath("sc_features")
        self.n_feat_dir = self.data.joinpath("nsc_features")

        self.d_feats = [str(path) for path in self.d_feat_dir.iterdir()]
        self.s_feats = [str(path) for path in self.s_feat_dir.iterdir()]
        self.n_feats = [str(path) for path in self.n_feat_dir.iterdir()]
        self.d_feats.sort()
        self.s_feats.sort()
        self.n_feats.sort()

        assert (
            len(self.d_feats) == len(self.s_feats) == len(self.n_feats)
        ), "The number of features are different from each other!"

        # image paths
        self.d_img_dir = self.data.joinpath("images")
        self.s_img_dir = self.data.joinpath("sc_images")
        self.n_img_dir = self.data.joinpath("nsc_images")

        self.d_imgs = [str(path) for path in self.d_img_dir.iterdir()]
        self.s_imgs = [str(path) for path in self.s_img_dir.iterdir()]
        self.n_imgs = [str(path) for path in self.n_img_dir.iterdir()]
        self.d_imgs.sort()
        self.s_imgs.sort()
        self.n_imgs.sort()

        # batch size and sentence number
        self.batch_size = batch_size
        self.seq_per_img = seq_per_img

        # splits
        with self.data.joinpath("splits.json").open("r") as f:
            self.splits = json.load(f)
        self.split = split
        if split == "train":
            self.split_idxs = self.splits["train"]
            self.num_samples = len(self.split_idxs)
        elif split == "val":
            self.split_idxs = self.splits["val"]
            self.num_samples = len(self.split_idxs)
        elif split == "test":
            self.split_idxs = self.splits["test"]
            self.num_samples = len(self.split_idxs)
        else:
            raise Exception("Unknown data split %s" % split)

        print("Dataset size for %s: %d" % (split, self.num_samples))

        # load in the sequence data
        self.h5_label_file = h5py.File(self.data.joinpath("labels.h5"), "r")
        seq_size = self.h5_label_file["labels"].shape
        self.labels = self.h5_label_file["labels"][:]  # just gonna load...
        self.neg_labels = self.h5_label_file["neg_labels"][:]
        self.max_seq_length = seq_size[1]
        self.label_start_idx = self.h5_label_file["label_start_idx"][:]
        self.label_end_idx = self.h5_label_file["label_end_idx"][:]
        self.neg_label_start_idx = self.h5_label_file["neg_label_start_idx"][:]
        self.neg_label_end_idx = self.h5_label_file["neg_label_end_idx"][:]
        print("Max sequence length is %d" % self.max_seq_length)
        self.h5_label_file.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        random.seed()
        img_idx = self.split_idxs[index]

        # Fetch image data
        # one easy way to augment data is to use nonsemantically changed
        # scene as the default :)
        if self.split == "train":
            if random.random() < 0.5:
                d_feat_path = self.d_feats[img_idx]
                d_img_path = self.d_imgs[img_idx]
                n_feat_path = self.n_feats[img_idx]
                n_img_path = self.n_imgs[img_idx]
            else:
                d_feat_path = self.n_feats[img_idx]
                d_img_path = self.n_imgs[img_idx]
                n_feat_path = self.d_feats[img_idx]
                n_img_path = self.d_imgs[img_idx]
        else:
            d_feat_path = self.d_feats[img_idx]
            d_img_path = self.d_imgs[img_idx]
            n_feat_path = self.n_feats[img_idx]
            n_img_path = self.n_imgs[img_idx]

        q_feat_path = self.s_feats[img_idx]
        q_img_path = self.s_imgs[img_idx]

        d_feature = torch.as_tensor(np.load(d_feat_path), dtype=torch.float32)
        n_feature = torch.as_tensor(np.load(n_feat_path), dtype=torch.float32)
        q_feature = torch.as_tensor(np.load(q_feat_path), dtype=torch.float32)

        # Fetch change type labels
        aux_label_pos = -1
        for type, img_set in self.type_to_img.items():
            if img_idx in img_set:
                aux_label_pos = self.type_to_label[type]
                break
        aux_label_neg = self.type_to_label["no_change"]

        # Fetch sequence labels
        ix1 = self.label_start_idx[img_idx]
        ix2 = self.label_end_idx[img_idx]
        n_cap = ix2 - ix1 + 1

        seq = np.zeros([self.seq_per_img, self.max_seq_length + 1], dtype=int)
        if n_cap < self.seq_per_img:
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, : self.max_seq_length] = self.labels[ixl, : self.max_seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            seq[:, : self.max_seq_length] = self.labels[
                ixl : ixl + self.seq_per_img, : self.max_seq_length
            ]
        seq = torch.as_tensor(seq, dtype=torch.int64)
        seq[torch.arange(self.seq_per_img), seq.argmin(1)] = self.word_to_idx["<END>"]

        # Fetch negative sequence labels
        ix1 = self.neg_label_start_idx[img_idx]
        ix2 = self.neg_label_end_idx[img_idx]
        n_cap = ix2 - ix1 + 1

        neg_seq = np.zeros([self.seq_per_img, self.max_seq_length + 1], dtype=int)
        if n_cap < self.seq_per_img:
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                neg_seq[q, : self.max_seq_length] = self.neg_labels[ixl, : self.max_seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            neg_seq[:, : self.max_seq_length] = self.neg_labels[
                ixl : ixl + self.seq_per_img, : self.max_seq_length
            ]
        neg_seq = torch.as_tensor(neg_seq, dtype=torch.int64)
        neg_seq[torch.arange(self.seq_per_img), neg_seq.argmin(1)] = self.word_to_idx["<END>"]

        # Generate masks
        mask = np.zeros_like(seq)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, seq)))
        for ix, row in enumerate(mask):
            row[: nonzeros[ix]] = 1
        mask = torch.as_tensor(mask)

        neg_mask = np.zeros_like(neg_seq)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, neg_seq)))
        for ix, row in enumerate(neg_mask):
            row[: nonzeros[ix]] = 1
        neg_mask = torch.as_tensor(neg_mask)

        return (
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
        )

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length


def collate_fn(batch):
    transposed = list(zip(*batch))
    d_feat_batch = transposed[0]
    n_feat_batch = transposed[1]
    q_feat_batch = transposed[2]
    seq_batch = default_collate(transposed[3])
    neg_seq_batch = default_collate(transposed[4])
    mask_batch = default_collate(transposed[5])
    neg_mask_batch = default_collate(transposed[6])
    aux_label_pos_batch = default_collate(transposed[7])
    aux_label_neg_batch = default_collate(transposed[8])
    if any(f is not None for f in d_feat_batch):
        d_feat_batch = default_collate(d_feat_batch)
    if any(f is not None for f in n_feat_batch):
        n_feat_batch = default_collate(n_feat_batch)
    if any(f is not None for f in q_feat_batch):
        q_feat_batch = default_collate(q_feat_batch)

    d_img_batch = transposed[9]
    n_img_batch = transposed[10]
    q_img_batch = transposed[11]
    return (
        d_feat_batch,
        n_feat_batch,
        q_feat_batch,
        seq_batch,
        neg_seq_batch,
        mask_batch,
        neg_mask_batch,
        aux_label_pos_batch,
        aux_label_neg_batch,
        d_img_batch,
        n_img_batch,
        q_img_batch,
    )


class RCCDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs["collate_fn"] = collate_fn
        super().__init__(dataset, **kwargs)


def create_dataset(
    data: Union[str, Path],
    split: str = "train",
    batch_size: int = 128,
    seq_per_img: int = 1,
    num_workers: int = 4,
) -> Tuple[Dataset, DataLoader]:

    if split == "train":
        shuffle = True
    else:
        shuffle = False

    dataset = RCCDataset(data, split, batch_size, seq_per_img)
    data_loader = RCCDataLoader(
        dataset, batch_size=dataset.batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataset, data_loader
