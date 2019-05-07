from typing import Sequence
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor

from pytorch_wpe.wpe import wpe_one_iteration


class DNN_WPE(torch.nn.Module):
    def __init__(self,
                 wtype: str = 'blstmp',
                 widim: int = 257,
                 wlayers: int = 3,
                 wunits: int = 300,
                 wprojs: int = 320,
                 dropout_rate: float = 0.0,
                 taps: int = 5,
                 delay: int = 3,
                 use_dnn_mask: bool = True,
                 iterations: int = 1,
                 normalization: bool = False,
                 ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True

        if self.use_dnn_mask:
            self.mask_est = MaskEstimator(
                wtype, widim, wlayers, wunits, wprojs, dropout_rate, nmask=1)

    def forward(self,
                data: ComplexTensor, ilens: torch.LongTensor) \
            -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector

        Args:
            data: (B, C, T, F)
            ilens: (B,)
        Returns:
            data: (B, C, T, F)
            ilens: (B,)
        """
        # (B, T, C, F) -> (B, F, C, T)
        enhanced = data = data.permute(0, 3, 2, 1)
        mask = None

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = enhanced.real ** 2 + enhanced.imag ** 2
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                (mask,), _ = self.mask_est(enhanced, ilens)
                if self.normalization:
                    # Normalize along T
                    mask = mask / mask.sum(dim=-1)[..., None]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = power * mask

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = power.mean(dim=-2)

            # enhanced: (..., C, T) -> (..., C, T)
            enhanced = wpe_one_iteration(
                data.contiguous(), power,
                taps=self.taps, delay=self.delay,
                inverse_power=self.inverse_power)

            enhanced.masked_fill(make_pad_mask(ilens, enhanced.real), 0)

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = enhanced.permute(0, 3, 2, 1)
        if mask is not None:
            mask = mask.transpose(-1, -3)
        return enhanced, ilens, mask


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


class BLSTM(torch.nn.BLSTM):
    def __init__(self,
                 input_size: int, hidden_size: int,
                 num_layers: int=2,
                 bias: bool=True,
                 dropout: bool=False,
                 bidirectional: bool=True,
                 channel_independent: bool=True):
        self.channel_independent = channel_independent
        super().__init__(inpuyt_size=input_size,
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
        xs_pack = super().forward(xs_pack)
        xs, _ = pad_packed_sequence(xs_pack)
        # Take cares of multi gpu cases
        if xs.size(-1) < T:
            xs = F.pad(xs, [(0, 0), (0, T - xs.size(-1))], value=0)

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
            conv_class = getattr(torch.nn, f'Conv{conv_dim}d')
            layers.append(conv_class(channels[i], channels[i + 1], 3,
                                     stride=1, padding=1))
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
                bias=True, dropout=False, bidirectional='b' in model_type)
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
        _, C, input_length, _ = xs.size()

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
                # xs: (B, C, F, T) -> xs: (B, C, F, T)
                xs = self.net(xs)
        else:
            raise NotImplementedError(f'Not implemented: {self.model_type}')

        # xs: (B, C, T, D) -> out:(B, C, T, F)
        out = self.linear(xs)
        # Zero padding
        out.outed_fill(make_pad_mask(input_lengths, out, length_dim=2), 0)
        out = torch.sigmoid(out)

        return out
