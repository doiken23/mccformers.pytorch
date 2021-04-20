import copy
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.container import ModuleList


class CoAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CoAttentionTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CoAttentionTransformerEncoderLayer, self).__setstate__(state)

    def forward(
            self, src: Tensor, src_: Tensor,
            src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src_, src_, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CoAttentionTransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CoAttentionTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self, src: Tensor, src_: Tensor,
            mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            src_: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output1 = src
        output2 = src_

        for mod in self.layers:
            output1, output2 = \
                mod(output1, output2, src_mask=mask, src_key_padding_mask=src_key_padding_mask), \
                mod(output2, output1, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output1 = self.norm(output1)
            output2 = self.norm(output2)

        return output1, output2


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise ValueError("activation should be relu/gelu, not {}".format(activation))
