from typing import Sequence, Tuple, Optional

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch_complex.tensor import ComplexTensor

from pytorch_wpe import wpe_one_iteration


class DNN_WPE(torch.nn.Module):
    def __init__(self,
                 model_type: str = 'blstm',
                 feat_type: str = 'log_power',
                 out_type: str = 'mask',
                 input_size: int = 257, hidden_size: int = 300,
                 num_layers: int = 2, nchannel: int = 8,
                 channel_independent: bool = True,
                 taps: int = 5, delay: int = 3, use_dnn: bool = True,
                 iterations: int = 1, normalization: bool = False,
                 ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay

        self.normalization = normalization
        self.use_dnn = use_dnn
        self.inverse_power = True
        if out_type is None:
            self.out_type = feat_type
        else:
            self.out_type = out_type

        if use_dnn:
            self.estimator = Estimator(
                 model_type=model_type, feat_type=feat_type,
                 input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers, nchannel=nchannel,
                 channel_independent=channel_independent)
        else:
            self.estimator = None

    def forward(self,
                data: ComplexTensor, ilens: torch.LongTensor=None,
                return_wpe: bool=True) -> Tuple[Optional[ComplexTensor],
                                                torch.Tensor]:
        if ilens is None:
            ilens = torch.full((data.size(0),), data.size(2),
                               dtype=torch.long, device=data.device)
        # data: (B, C, T, F), ilens: (B,)
        enhanced = data
        power = None
        for i in range(self.iterations):
            # Calculate power: (B, C, T, F)
            power = enhanced.real ** 2 + enhanced.imag ** 2
            if i == 0 and self.use_dnn:
                # mask: (B, C, T, F)
                mask = self.estimator(enhanced, ilens)
                if self.normalization:
                    # Normalize along T
                    mask = mask / mask.sum(dim=-2)[..., None]
                if self.out_type == 'mask':
                    power = power * mask
                else:
                    power = mask

                    if self.out_type == 'amplitude':
                        power = power ** 2
                    elif self.out_type == 'log_power':
                        power = power.exp()
                    elif self.out_type == 'power':
                        pass
                    else:
                        raise NotImplementedError(self.out_type)

            if not return_wpe:
                return None, power

            # power: (B, C, T, F) -> _power: (B, F, T)
            _power = power.mean(dim=1).transpose(-1, -2).contiguous()

            # data: (B, C, T, F) -> _data: (B, F, C, T)
            _data = data.permute(0, 3, 1, 2).contiguous()
            # _enhanced: (B, F, C, T)
            _enhanced_real = []
            _enhanced_imag = []
            for d, p, l in zip(_data, _power, ilens):
                # e: (F, C, T) -> (T, C, F)
                e = wpe_one_iteration(
                    d[..., :l], p[..., :l],
                    taps=self.taps, delay=self.delay,
                    inverse_power=self.inverse_power).transpose(0, 2)
                _enhanced_real.append(e.real)
                _enhanced_imag.append(e.imag)
            # _enhanced: B x (T, C, F) -> (B, T, C, F) -> (B, F, C, T)
            _enhanced_real = pad_sequence(_enhanced_real,
                                          batch_first=True).transpose(1, 3)
            _enhanced_imag = pad_sequence(_enhanced_imag,
                                          batch_first=True).transpose(1, 3)
            _enhanced = ComplexTensor(_enhanced_real, _enhanced_imag)

            # enhanced: (B, F, C, T) -> (B, C, T, F)
            enhanced = _enhanced.permute(0, 2, 3, 1)

        # enhanced: (B, C, T, F), power: (B, C, T, F)
        return enhanced, power


def make_pad_mask(lengths, xs=None, length_dim=-1):
    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None
                    for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


class BLSTM(torch.nn.LSTM):
    def __init__(self,
                 input_size: int, hidden_size: int,
                 num_layers: int = 2,
                 bias: bool = True,
                 dropout: float = 0.,
                 bidirectional: bool = True,
                 channel_independent: bool = True):
        self.channel_independent = channel_independent
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         bias=bias,
                         dropout=dropout,
                         bidirectional=bidirectional)

    def forward(self, xs: torch.Tensor, input_lengths: torch.LongTensor):
        # xs: (B, C, T, F)
        B, C, T, F = xs.size()
        if self.channel_independent:
            # xs: (B, C, T, F) -> xs: (B * C, 1, T, F)
            xs = xs.view(-1, 1, T, F)
            # input_lengths: (B,) -> input_lengths_: (B * C)
            input_lengths = \
                input_lengths[:, None].expand(-1, C).contiguous().view(-1)

        # xs: (B, C, T, F) -> xs: (B, T, C * F)
        xs = xs.transpose(1, 2).contiguous().view(xs.size(0), T, -1)

        xs_pack = pack_padded_sequence(xs, input_lengths, batch_first=True)
        xs_pack, _ = super().forward(xs_pack)
        xs, _ = pad_packed_sequence(xs_pack, batch_first=True, total_length=T)

        if self.channel_independent:
            # xs: (B * C, 1, T, F) -> xs: (B, C, T, F)
            xs = xs.view(B, C, T, -1)
        else:
            # xs: (B, T, C * F) -> xs: (B, C, T, F)
            xs = xs.view(B, T, C, -1).transpose(1, 2)

        # xs: (B, C, T, F)
        return xs


class CNN(torch.nn.Sequential):
    def __init__(self, channels: Sequence[int]=(8, 64, 64, 8),
                 conv_dim: int=2):
        layers = []
        for i in range(len(channels) - 1):
            Convnd = getattr(torch.nn, f'Conv{conv_dim}d')
            layers.append(Convnd(channels[i], channels[i + 1], 3, stride=1,
                                 padding=1))
            layers.append(torch.nn.ReLU)
        super().__init__(layers)


class Estimator(torch.nn.Module):
    def __init__(self, model_type, feat_type: str = 'amplitude',
                 input_size: int=400, hidden_size: int=1024,
                 num_layers: int=2, nchannel: int=8,
                 channel_independent: bool=True):
        super().__init__()

        self.channel_independent = channel_independent
        supported = ('amplitude', 'power', 'log_power', 'concat')
        if feat_type not in supported:
            raise ValueError(
                f'feat_type must be one of {supported}: {feat_type} ')
        self.feat_type = feat_type

        self.model_type = model_type
        if model_type in ('blstm', 'lstm'):
            self.net = BLSTM(
                input_size=input_size
                if channel_independent else nchannel * input_size,
                channel_independent=channel_independent,
                hidden_size=hidden_size, num_layers=num_layers,
                bias=True, dropout=0, bidirectional='b' in model_type)
            outsize = (2 if 'b' in model_type else 1) * hidden_size

        elif model_type == 'cnn':
            if channel_independent:
                # in: (B * C, F, T)
                channels = [input_size] + [hidden_size
                                           for _ in range(num_layers)]
                self.net = CNN(channels, conv_dim=1)
                outsize = hidden_size
            else:
                # in: (B, C, T, F)
                channels = [nchannel] + \
                    [hidden_size for _ in range(num_layers - 1)] + [nchannel]
                self.net = CNN(channels, conv_dim=2)
                outsize = input_size
        else:
            raise NotImplementedError(model_type)

        self.linear = torch.nn.Linear(outsize, input_size)

    def forward(self, xs: ComplexTensor, input_lengths: torch.LongTensor) \
            -> torch.Tensor:
        assert xs.size(0) == input_lengths.size(0), (xs.size(0),
                                                     input_lengths.size(0))

        # xs: (B, C, T, D)
        _, C, _, _ = xs.size()

        if self.feat_type == 'amplitude':
            # xs: (B, C, T, F) -> (B, C, T, F)
            xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        elif self.feat_type == 'power':
            # xs: (B, C, T, F) -> (B, C, T, F)
            xs = xs.real ** 2 + xs.imag ** 2
        elif self.feat_type == 'log_power':
            # xs: (B, C, T, F) -> (B, C, T, F)
            xs = torch.log(xs.real ** 2 + xs.imag ** 2)
        elif self.feat_type == 'concat':
            # xs: (B, C, T, F) -> (B, C, T, 2 * F)
            xs = torch.cat([xs.real, xs.imag], -1)
        else:
            raise NotImplementedError(f'Not implemented: {self.feat_type}')

        if self.model_type in ('blstm', 'lstm'):
            # xs: (B, C, T, F) -> xs: (B, C, T, D)
            xs = self.net(xs, input_lengths)

        elif self.model_type == 'cnn':
            if self.channel_independent:
                # xs: (B, C, T, F) -> xs: (B * C, F, T)
                xs = xs.view(-1, xs.size(2), xs.size(3)).transpose(2, 3)
                # xs: (B * C, F, T) -> xs: (B * C, D, T)
                xs = self.net(xs)
                # xs: (B * C, D, T) -> (B, C, T, D)
                xs = xs.transpose(2, 3).contiguous().view(
                    -1, C, xs.size(3), xs.size(2))
            else:
                # xs: (B, C, T, F) -> xs: (B, C, T, F)
                xs = self.net(xs)
        else:
            raise NotImplementedError(f'Not implemented: {self.model_type}')

        # xs: (B, C, T, D) -> out:(B, C, T, F)
        out = self.linear(xs)
        # Zero padding
        out.masked_fill(make_pad_mask(input_lengths, out, length_dim=2), 0)
        out = torch.sigmoid(out)

        return out
