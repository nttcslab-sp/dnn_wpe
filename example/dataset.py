import collections
from typing import Tuple, List, Union, Sequence

import numpy
import scipy.io.wavfile as scipy_wav
import scipy.signal
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch_complex.tensor import ComplexTensor
import torch_complex.functional as FC
from typeguard import check_argument_types, typechecked

from utils import SoundScpReader
from utils import Stft


class ScpScpDataset:
    """
    >>> d = ScpScpDataset('data/SimData_dt_for_8ch_far_room1/reverb.scp',
                          'data/SimData_dt_for_8ch_far_room1/clean.scp')
    >>> key, x_stft, t_stft, _ = d[0]

    """

    def __init__(self, x_scp: str, t_scp: str, stft_conf: str = None):
        self.x_reader = SoundScpReader(x_scp)
        self.t_reader = SoundScpReader(t_scp)
        self.keys = list(self.x_reader.keys())
        self.stft_func = Stft(stft_conf)

    def __len__(self):
        return len(self.x_reader)

    def __getitem__(self, key: Union[str, int]) \
            -> Tuple[str, numpy.ndarray, numpy.ndarray]:
        if isinstance(key, int):
            key = self.keys[key]

        x_rate, x = self.x_reader[key]
        t_rate, t = self.t_reader[key]

        if x.ndim == 1:
            x = x[:, None]
        if t.ndim == 1:
            t = t[:, None]
        assert x_rate == t_rate, (x_rate, t_rate)
        # x, t: (T, C) -> (C, T)
        x = x.T
        t = t.T

        x = x.astype(numpy.double)
        t = t.astype(numpy.double)

        # Scaling to -1-1
        x = x / (numpy.iinfo(numpy.int16).max - 1)
        t = t / (numpy.iinfo(numpy.int16).max - 1)

        # Stft in double precision and cast to float
        # x: (C, T) -> x_stft: (C, F, T)
        x_stft = self.stft_func(x).astype(numpy.complex64)
        # t: (C, T,) -> t_stft: (C, F, T)
        t_stft = self.stft_func(t).astype(numpy.complex64)

        # Note: Reverberant data is longer than than the original signal.
        x_stft = x_stft[..., :t_stft.shape[-1]]

        # x_stft: (C, F, T) -> (C, T, F)
        x_stft = x_stft.transpose(0, 2, 1)
        # t_stft: (F, T) -> (T, F)
        t_stft = t_stft.transpose(0, 2, 1)

        return key, x_stft, t_stft


class WavRIRNoiseDataset:
    def __init__(self, wav_rir_noise_scp: str, stft_conf: str = None,
                 norm_scale: bool = True, delay: int = 0):
        self.reader = WavRIRNoiseReader(wav_rir_noise_scp,
                                        norm_scale=norm_scale,
                                        delay=delay)
        self.keys = list(self.reader.keys())
        self.stft_func = Stft(stft_conf)

    def __len__(self):
        return len(self.reader)

    @typechecked
    def __getitem__(self, key: Union[str, int]) \
            -> Tuple[str, numpy.ndarray, numpy.ndarray]:
        if isinstance(key, int):
            key = self.keys[key]

        x, t = self.reader[key]
        # x, t: (T, C) -> (C, T)
        x = x.T
        t = t.T

        # Stft in double precision and cast to float
        # x: (C, T) -> x_stft: (C, F, T)
        x_stft = self.stft_func(x).astype(numpy.complex64)
        # t: (C, T,) -> t_stft: (C, F, T)
        t_stft = self.stft_func(t).astype(numpy.complex64)

        # Note: Reverberant data is longer than than the original signal.
        x_stft = x_stft[..., :t_stft.shape[-1]]

        # x_stft: (C, F, T) -> (C, T, F)
        x_stft = x_stft.transpose(0, 2, 1)
        # t_stft: (F, T) -> (T, F)
        t_stft = t_stft.transpose(0, 2, 1)
        return key, x_stft, t_stft


class DescendingOrderedBatchSampler:
    def __init__(self, nframes: str,
                 batch_size: int=1, shuffle: bool = False,
                 drop_last: bool = True):
        assert check_argument_types()
        with open(nframes) as f:
            def _func(s):
                return s[0], int(s[1])
            # Sort by the signal length
            self.nframes = sorted((_func(l.split(maxsplit=1)) for l in f),
                                  key=lambda x: -x[1])

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self) -> List[str]:
        if self.shuffle:
            indices = torch.randperm(len(self))
        else:
            indices = range(len(self))

        for idx in indices:
            yield [k for k, _ in self.nframes[idx * self.batch_size:
                                              (idx + 1) * self.batch_size]]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.nframes) // self.batch_size
        else:
            return (len(self.nframes) + self.batch_size - 1) // self.batch_size


class CollateFuncWithDevice:
    def __init__(self, device, collate_fn):
        self.device = device
        self.collate_fn = collate_fn

    def __call__(self, data):
        return self.to(self.collate_fn(data))

    def to(self, vs):
        if isinstance(vs, dict):
            return {k: self.to(v) for k, v in value.items()}
        elif isinstance(vs, (list, tuple)):
            return type(vs)(self.to(v) for v in vs)
        else:
            return vs.to(self.device) if torch.is_tensor(vs) else vs


@typechecked
def collate_fn(data: Sequence[Tuple[str, numpy.ndarray, numpy.ndarray]]) \
        -> Tuple[Tuple[str, ...], ComplexTensor, ComplexTensor,
                 torch.LongTensor]:

    # Check shape:
    for k, x, t in data:
        assert isinstance(k, str), type(k)
        # Expected: x: (C, T, F), t: (C, T, F)
        assert x.ndim == 3, x.shape
        assert x.shape == t.shape, (x.shape, t.shape)


    # Create input lengths
    ilens = numpy.array([x.shape[-2] for k, x, t in data], dtype=numpy.long)
    ilens = torch.from_numpy(ilens)

    # Sort by the input length
    ilens, index = torch.sort(ilens, descending=True)
    data = [data[i] for i in index]

    # From numpy to torch Tensor:
    # x: (C, T, F) -> (T, C, F)
    xs_real = [torch.from_numpy(x.real).transpose(0, 1) for k, x, t in data]
    xs_imag = [torch.from_numpy(x.imag).transpose(0, 1) for k, x, t in data]
    # t: (C, T, F) -> (T, C, F)
    ts_real = [torch.from_numpy(t.real).transpose(0, 1) for k, x, t in data]
    ts_imag = [torch.from_numpy(t.imag).transpose(0, 1) for k, x, t in data]

    # Zero padding
    # xs: B x (T, C, F) -> (B, T, C, F) -> (B, C, T, F)
    xs_real = pad_sequence(xs_real, batch_first=True).transpose(1, 2)
    xs_imag = pad_sequence(xs_imag, batch_first=True).transpose(1, 2)
    # ts: B x (T, C, F) -> (B, T, C, F) -> (B, C, T, F)
    ts_real = pad_sequence(ts_real, batch_first=True).transpose(1, 2)
    ts_imag = pad_sequence(ts_imag, batch_first=True).transpose(1, 2)

    xs = ComplexTensor(xs_real, xs_imag)
    ts = ComplexTensor(ts_real, ts_imag)

    # xs: (B, C, T, F), ts: (B, C, T, F), ilens: (B,)
    ks = tuple(k for k, x, t in data)
    return ks, xs, ts, ilens


class Chunk:
    def __init__(self, dataloader: DataLoader, batch_size: int,
                 width: int = 40,
                 lcontext: int = 4, rcontext: int = 4):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.width = width
        self.lcontext = lcontext
        self.rcontext = rcontext

    def __iter__(self):
        _ks = []
        _xs = []
        _ts = []
        _ilens = []

        for ks, xs, ts, ilens in self.dataloader:
            assert len(ks) == 1, \
                f'batch-size of dataloder is not 1: {len(ks)} != 1'
            # Check shape:
            assert isinstance(ks[0], str), type(ks[0])
            # Expected: x: ( C, T, F), t: (C, T, F)
            assert xs[0].dim() == 3, xs[0].shape
            assert xs[0].shape == ts[0].shape, (xs[0].shape, ts[0].shape)

            offset = 0
            while True:
                ilen = ilens[0]
                if offset + self.width > ilen:
                    break

                _k = ks[0]
                _x = xs[0, :,
                        max(offset - self.lcontext, 0):offset + self.width + self.rcontext, :]
                if _x.shape[1] < self.width + self.lcontext + self.rcontext:
                    lp = max(0, self.lcontext - offset)
                    rp = max(offset + self.width + self.rcontext - xs.size(2), 0)
                    _x = FC.pad(_x, (0, 0, rp, lp, 0, 0), mode='constant')

                # _t: (C, width, F)
                _t = ts[0, :, offset:offset + self.width, :]
                _l = self.width + self.lcontext + self.rcontext

                _ks.append(_k)
                _xs.append(_x)
                _ts.append(_t)
                _ilens.append(_l)
                offset += self.width

                if len(_ks) == self.batch_size:
                    _ks = tuple(_ks)
                    # _x: (C, width, context, F)
                    yield _ks, FC.stack(_xs), \
                        FC.stack(_ts), torch.tensor(_ilens, device=_xs[0].device)

                    _ks = []
                    _xs = []
                    _ts = []
                    _ilens = []


class RIRandNoisePairing:
    def __init__(self, rir_list: str, noise_list: str):
        self.rir_list = rir_list
        self.noise_list = noise_list

        # Format or rir and noise list
        #     roomid1 a.wav b.wav c.wav
        #     roomid2 d.wav
        #     roomid3 e.wav f.wav

        with open(rir_list, 'r') as f:
            self.room2rirs = collections.OrderedDict()
            for line in f:
                sps = line.rstrip().split()
                self.room2rirs[sps[0]] = sps[1:]

        with open(noise_list, 'r') as f:
            self.room2noises = collections.OrderedDict()
            for line in f:
                sps = line.rstrip().split()
                self.room2noises[sps[0]] = sps[1:]

        if set(self.room2noises) != set(self.room2rirs):
            raise RuntimeError(f'Mismatched ids: {rir_list} and {noise_list}')
        self.rooms = list(self.room2rirs)

    def __iter__(self):
        # Generate RIR and Noise pair cyclically
        room2noise_idx = {r: 0 for r in self.rooms}
        while True:
            for room in self.rooms:
                for rir in self.room2rirs[room]:
                    noise_idx = room2noise_idx[room]
                    if noise_idx == (len(self.room2noises[room]) - 1):
                        room2noise_idx[room] = 0
                    else:
                        room2noise_idx[room] += 1
                    yield rir, self.room2noises[room][noise_idx]


def paring_wav_rir_noise(wav_scp, rir_list: str, noise_list: str,
                         out_file: str,
                         lower_snrdb: float = 10, upper_snrdb: float = 30,
                         snrdb_seed: int = 0):
    it = iter(RIRandNoisePairing(rir_list, noise_list))
    state = numpy.random.RandomState(snrdb_seed)

    with open(wav_scp, 'r') as f, open(out_file, 'w') as fo:
        # In Format
        #    uttid1 a.wav
        #    uttid2 b.wav
        # Out Format
        #    uttid1 a.wav rir1.wav noise1.wav 12.3
        #    uttid2 b.wav rir2.wav noise2.wav 20.8
        for line in f:
            uid, wav = line.rstrip().split()
            rir, noise = next(it)
            snrdb = state.uniform(lower_snrdb, upper_snrdb)
            fo.write(f'{uid} {wav} {rir} {noise} {snrdb}\n')


class WavRIRNoiseReader:
    def __init__(self, wav_rir_noise_scp: str, predelay_ms: float=50,
                 norm_scale: bool=True,
                 delay=0):
        # wav_rir_noise_scp can be created by paring_wav_rir_noise
        with open(wav_rir_noise_scp, 'r') as f:
            self.id2paths = {}
            for line in f:
                uid, wav_path, rir_path, noise_path, snrdb \
                    = line.rstrip().split()
                self.id2paths[uid] = wav_path, rir_path, noise_path, snrdb

        self._keys = tuple(self.id2paths)
        self.predelay_ms = predelay_ms
        self.delay = delay
        self.norm_scale = norm_scale

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys

    def __getitem__(self, item: Union[str, int]) \
            -> Tuple[numpy.ndarray, numpy.ndarray]:
        if isinstance(item, int):
            item = self._keys[item]

        wav_path, rir_path, noise_path, snrdb = self.id2paths[item]
        snrdb = float(snrdb)

        rate, speech = scipy_wav.read(wav_path)
        assert speech.ndim == 1, f'Must be single channel: {speech.ndim} != 1'

        _r, rir = scipy_wav.read(rir_path)
        assert rate == _r, (rate, _r)
        if rir.ndim == 1:
            rir = rir[:, None]
        # (Time, Nmic) -> (Nmic, Time)
        rir = rir.T

        _r, noise = scipy_wav.read(noise_path)
        assert rate == _r, (rate, _r)
        if noise.ndim == 1:
            noise = noise[:, None]
        # (Time, Nmic) -> (Nmic, Time)
        noise = noise[:len(speech), :].T
        assert rir.shape[0] == noise.shape[0], \
            f'Channels mismatch: {rir.shape[0]} != {noise.shape[0]}'

        # Scaling to [-1., 1.]
        speech = \
            speech.astype(numpy.float64) / (numpy.iinfo(speech.dtype).max - 1)
        rir = rir.astype(numpy.float64) / (numpy.iinfo(rir.dtype).max - 1)
        noise = \
            noise.astype(numpy.float64) / (numpy.iinfo(noise.dtype).max - 1)

        # 1. Create reverberant signal
        reverb = scipy.signal.convolve(
            speech[None], rir, mode='full')[:, :len(speech)]

        # 2. Create direct + early reflection
        # The arrival time of direct signal
        dt = numpy.argmax(rir, axis=1).min()
        # The arrival time of early reflection
        et = dt + int(self.predelay_ms * rate / 1000)
        rir_direct = rir[:, :et]

        direct = scipy.signal.convolve(
            speech[None], rir_direct, mode='full')[:, :len(speech)]

        # 3. Add noise
        # SNR:= direct_power / noise_power
        noise = (noise * numpy.sqrt((direct ** 2).mean()) /
                 numpy.sqrt((noise ** 2).mean()) *
                 10 ** (-snrdb / 20))

        if self.delay != 0:
            reverb = reverb[:, self.delay:] + noise[:, self.delay:]
            direct = direct[:, :-self.delay] + noise[:, :-self.delay]
        else:
            reverb = reverb + noise
            direct = direct + noise

        if self.norm_scale:
            m = numpy.abs(reverb).mean()
            reverb /= m
            direct /= m

        return reverb.T, direct.T
