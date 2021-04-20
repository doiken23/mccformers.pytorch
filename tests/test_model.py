#!/usr/bin/env python3

import pytest

import torch

from models import MCCFormer
from models.model import MCCFormerEncoderD, MCCFormerEncoderS


@pytest.fixture
def parameters():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    before_img = torch.randn(4, 1024, 14, 14, device=device)
    after_img = torch.randn(4, 1024, 14, 14, device=device)

    targets = torch.zeros(100, 4, dtype=torch.int64, device=device)

    return device, before_img, after_img, targets


def test_encoder_d(parameters):
    device, before_img, after_img, _ = parameters

    model = MCCFormerEncoderD(1024, 512)
    model.to(device)

    outputs = model(before_img, after_img)

    assert outputs.size() == (14 ** 2, 4, 2 * 512)


def test_encoder_s(parameters):
    device, before_img, after_img, _ = parameters

    model = MCCFormerEncoderS(1024, 512)
    model.to(device)

    outputs = model(before_img, after_img)

    assert outputs.size() == (14 ** 2, 4, 2 * 512)


def test_mccformer(parameters):
    device, before_img, after_img, targets = parameters

    model = MCCFormer('D', 1000, max_len=100)
    model.to(device)

    # check train model
    model.train()
    loss = model(before_img, after_img, targets)
    loss.backward()
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32

    # check eval model
    model.eval()
    outputs = model(before_img, after_img)
    assert isinstance(loss, torch.Tensor)
    assert outputs.dtype == torch.int64
    assert outputs.size() == (100, 4)
