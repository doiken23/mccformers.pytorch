import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.transformers import (CoAttentionTransformerEncoder,
                                 CoAttentionTransformerEncoderLayer)


class PositionalEncoding(nn.Module):
    """
    Learnable position embeddings

    Args:
        pe_type (str): type of position embeddings,
            which is chosen from ['fully_learnable', 'sinusoidal']
        d_model (int): embed dim (required).
        max_len (int): max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model, max_len=100)
    """

    def __init__(self, pe_type: str, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        if pe_type == 'fully_learnable':
            self.pe = nn.parameter.Parameter(torch.randn(max_len, 1, d_model))
        elif pe_type == 'sinusoidal':
            # this part is copied from
            # https://github.com/pytorch/examples/blob/507493d7b5fab51d55af88c5df9eadceb144fb67/
            # word_language_model/model.py#L65-L106
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
        else:
            raise RuntimeError(
                'PE type should be fully_learnable/sinusoidal, not {}'.format(pe_type))

    def forward(self, x: Tensor) -> Tensor:
        """Inputs of forward function
        Args:
            x (Tensor): the sequence fed to the positional encoder model [L, N, C]
        Returns:
            output (Tensor): position embeddings [L, N, C]
        """
        return x + self.pe[:x.size(0)]


class SimpleEncoder(nn.Module):
    def __init__(self, feature_dim, encoder_dim, image_size=14):
        super(SimpleEncoder, self).__init__()

        self.linear = nn.Conv2d(feature_dim, encoder_dim, kernel_size=1)
        self.positional_encoding = PositionalEncoding('fully_learnable', 2 * encoder_dim, image_size ** 2)

    def forward(self, x1, x2):
        return self.positional_encoding(torch.cat([
            self.linear(x1).flatten(start_dim=2).permute(2, 0, 1),
            self.linear(x2).flatten(start_dim=2).permute(2, 0, 1)
        ], dim=2))


class MCCFormerEncoderD(nn.Module):
    def __init__(
            self, feature_dim: int, encoder_dim: int, feature_extractor: Optional[nn.Module] = None,
            image_size: int = 14, nhead: int = 4, transformer_layer_num: int = 2):
        super(MCCFormerEncoderD, self).__init__()

        # CNN
        self.feature_extractor = feature_extractor
        self.linear = nn.Conv2d(feature_dim, encoder_dim, kernel_size=1)

        # position embedding
        self.positional_encoding = PositionalEncoding('fully_learnable', encoder_dim, image_size ** 2)

        # Transformer
        encoder_layer = CoAttentionTransformerEncoderLayer(
            d_model=encoder_dim, nhead=nhead, dim_feedforward=4 * encoder_dim)
        self.transformer = CoAttentionTransformerEncoder(encoder_layer, num_layers=transformer_layer_num)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        N = len(x1)

        # extract visual features from before and after images
        features = torch.cat([x1, x2], dim=0)  # [2N, C, H, W]
        if self.feature_extractor is not None:
            features = self.feature_extractor(features)  # [2N, C, H', W']
        features = self.linear(features).flatten(start_dim=2).permute(2, 0, 1)  # [H'xW', 2N, C]

        # apply Transformer to before and after visual features
        before_features, after_features = features[:, :N], features[:, N:]  # [H'xW', N, C]
        before_features = self.positional_encoding(before_features)
        after_features = self.positional_encoding(after_features)
        before_features, after_features = self.transformer(before_features, after_features)

        return torch.cat([before_features, after_features], dim=2)  # [H'xW', N, 2C]


class MCCFormerEncoderS(nn.Module):
    def __init__(
            self, feature_dim: int, encoder_dim: int, feature_extractor: Optional[nn.Module] = None,
            image_size: int = 14, nhead: int = 4, transformer_layer_num: int = 2):
        super(MCCFormerEncoderS, self).__init__()

        self.image_size = image_size

        # CNN
        self.feature_extractor = feature_extractor
        self.linear = nn.Conv2d(feature_dim, encoder_dim, kernel_size=1)

        # position embedding
        self.positional_encoding = PositionalEncoding('fully_learnable', encoder_dim, 2 * image_size ** 2)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=nhead, dim_feedforward=4 * encoder_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layer_num)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        N = len(x1)

        # extract visual features from before and after images
        features = torch.cat([x1, x2], dim=0)  # [2N, 3, H, W]
        if self.feature_extractor is not None:
            features = self.feature_extractor(features)  # [2N, C, H', W']
        features = self.linear(features).flatten(start_dim=2).permute(2, 0, 1)  # [H'xW', 2N, C]
        features = torch.cat([features[:, :N], features[:, N:]], dim=0)  # [2xH'xW', N, C]
        features = self.positional_encoding(features)

        # apply Transformer to before and after visual features
        features = self.transformer(features)
        before_features, after_features = \
            features[:self.image_size ** 2], features[self.image_size ** 2:]  # [H'xW', N, C]

        return torch.cat([before_features, after_features], dim=2)  # [H'xW', N, 2C]


class MCCFormer(nn.Module):
    def __init__(
            self, encoder_type: str, num_tokens: int,
            # parameters of encoder
            feature_extractor: Optional[nn.Module] = None, feature_dim: int = 1024,
            encoder_dim: int = 512, image_size: int = 14,
            encoder_nhead: int = 4, encoder_transformer_layer_num: int = 2,
            # parameters of decoder
            decoder_nhead: int = 4, decoder_transformer_layer_num: int = 2,
            pe_type: str = 'fully_learnable', max_len=20):
        super(MCCFormer, self).__init__()

        # set encoder
        if encoder_type == 'D':
            self.encoder = MCCFormerEncoderD(
                feature_dim, encoder_dim, feature_extractor,
                image_size=image_size, nhead=encoder_nhead,
                transformer_layer_num=encoder_transformer_layer_num)
        elif encoder_type == 'S':
            self.encoder = MCCFormerEncoderS(
                feature_dim, encoder_dim, feature_extractor,
                image_size=image_size, nhead=encoder_nhead,
                transformer_layer_num=encoder_transformer_layer_num)
        elif encoder_type == 'Simple':
            self.encoder = SimpleEncoder(feature_dim, encoder_dim, image_size=image_size)
        else:
            raise RuntimeError('Invalid encoder type. Expect "D" or "S".')

        # set decoder
        decoder_dim = 2 * encoder_dim
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim, nhead=decoder_nhead, dim_feedforward=4 * decoder_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_transformer_layer_num)

        # position embeddings
        self.max_len = max_len
        self.positional_encoding = PositionalEncoding('fully_learnable', decoder_dim, max_len=max_len)

        # embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=decoder_dim, padding_idx=0)

        # caption predictor
        self.cap_predictor = nn.Linear(decoder_dim, num_tokens)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """
        This function is copied from
            https://github.com/pytorch/pytorch/blob/e61b4fa6915a9eaba7d1f86d3f7a3c3a763052e5/
            torch/nn/modules/transformer.py#L131-L137
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x1: Tensor, x2: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x1 (Tensor): before images [N, 3, H, W]
            x2 (Tensor): after images [N, 3, H, W]
            target (Optional[Tensor]): change captions [N, L]

        Returns:
            loss (Tensor): cross entropy loss
        """
        N = len(x1)
        device = x1.device
        assert x1.size() == x2.size()

        if self.training and target is None:
            raise RuntimeError('In the training mode, target should not be None.')

        if target is not None:
            target = target.transpose(0, 1)

        # encode visual features
        encoded_features = self.encoder(x1, x2)  # [H'xW', N, 2C]

        # decode change captions from visual features
        if self.training:
            # predict captions
            embeddings = self.positional_encoding(self.embedding_layer(target[:-1]))
            tgt_mask = self.generate_square_subsequent_mask(len(target) - 1).to(device)
            outputs = self.cap_predictor(self.decoder(embeddings, encoded_features, tgt_mask=tgt_mask))

            # compute loss
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(2)), target[1:].contiguous().view(-1), ignore_index=0)

            return loss
        else:
            if target is not None:
                embeddings = self.positional_encoding(self.embedding_layer(target[:-1]))
                tgt_mask = self.generate_square_subsequent_mask(len(target) - 1).to(device)
                outputs = self.cap_predictor(self.decoder(embeddings, encoded_features, tgt_mask=tgt_mask))
                _, outputs = outputs.max(2)
                return outputs.transpose(0, 1)
            else:
                # predict next words one-by-one manner
                outputs = torch.zeros(1, N, dtype=torch.int64, device=device) + 2
                for i in range(self.max_len - 1):
                    embeddings = self.positional_encoding(self.embedding_layer(outputs))
                    tgt_mask = self.generate_square_subsequent_mask(len(outputs)).to(device)
                    tmp_outputs = self.cap_predictor(self.decoder(embeddings, encoded_features, tgt_mask=tgt_mask))
                    _, predicted = tmp_outputs[-1].max(1)
                    outputs = torch.cat([outputs, predicted.unsqueeze(0)])

                return outputs.transpose(0, 1)
