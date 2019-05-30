import collections
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.io.wavfile as scipy_wav
import soundfile
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types


class Stft:
    def __init__(self, conf: str=None, fs: int=16000, window: str='hann',
                 frame_length: int=512, frame_shift: int=None, nfft: int=None,
                 detrend=False, return_onesided: bool=True,
                 boundary: str='zeros',
                 padded=True, axis=-1):
        assert check_argument_types()

        if frame_shift is None:
            frame_shift = frame_length // 2
        default_conf = dict(
            fs=fs, window=window, frame_length=frame_length,
            frame_shift=frame_shift,
            nfft=nfft, detrend=detrend, return_onesided=return_onesided,
            boundary=boundary, padded=padded, axis=axis)

        if conf is not None:
            with open(conf, 'r') as f:
                json_conf = json.load(f)
            if not isinstance(json_conf, dict):
                raise RuntimeError(
                    f'{conf} must be dict, but got {type(json_conf)}')

            for k in json_conf:
                if k not in default_conf:
                    raise RuntimeError(
                        f'Unknown option: {k}: '
                        f'must be one of {list(default_conf)}')
                else:
                    default_conf[k] = json_conf[k]
        self.conf = default_conf

    @property
    def frame_shift(self):
        return self.conf['frame_shift']

    @property
    def frame_length(self):
        return self.conf['frame_length']

    @property
    def fs(self):
        return self.conf['fs']

    def __call__(self, x: np.ndarray, **kwargs):
        return self.stft(x, **kwargs)

    def stft(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # x: (.,,, Time)
        d = self.conf.copy()
        d.update(kwargs)
        d['nperseg'] = d.pop('frame_length')
        d['noverlap'] = d['nperseg'] - d.pop('frame_shift')

        _, _, stft = scipy.signal.stft(x, **d)
        # stft: (..., Freq, Time)
        return stft

    def istft(self, x, **kwargs) -> np.ndarray:
        # x: (.,,, Freq, Time)
        d = self.conf.copy()
        d.update(kwargs)
        d['nperseg'] = d.pop('frame_length')
        d['noverlap'] = d['nperseg'] - d.pop('frame_shift')

        d.pop('detrend', None)
        d['input_onesided'] = d.pop('return_onesided')
        d.pop('padded', None)
        d.pop('axis', None)

        _, istft = scipy.signal.istft(x, **d)
        return istft


class SoundScpWriter:
    """

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

    >>> writer = SoundScpWriter('./data/', 'feat')
    >>> writer['aa'] = 16000, numpy_array
    >>> writer['bb'] = 16000, numpy_array

    """
    def __init__(self, basedir, name, format='wav', dtype=None):
        self.dir = Path(basedir) / f'data_{name}'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fscp = (Path(basedir) / f'{name}.scp').open('w')
        self.format = format
        self.dtype = dtype

    def __setitem__(self, key, value):
        rate, signal = value
        assert isinstance(rate, int), type(rate)
        assert isinstance(signal, np.ndarray), type(signal)
        wav = self.dir / f'{key}.{self.format}'
        wav.parent.mkdir(parents=True, exist_ok=True)
        if self.dtype is not None:
            signal = signal.astype(self.dtype)
        if self.format == 'wav':
            scipy_wav.write(wav, rate, signal)
        else:
            soundfile.write(str(wav), signal, rate)
        self.fscp.write(f'{key} {wav}\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


class SoundScpReader(collections.abc.Mapping):
    """

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

    >>> reader = SoundScpReader('wav.scp')
    >>> rate, array = reader['key1']

    """
    def __init__(self, fname, dtype=np.int16):
        self.fname = fname
        self.dtype = dtype
        with open(fname, 'r') as f:
            self.data = dict(line.rstrip().split(maxsplit=1) for line in f)

    def __getitem__(self, key):
        wav = self.data[key]
        if Path(wav).suffix == '.wav':
            # If wav format, use scipy.io.wavfile,
            # because sndfile always normalizes the data to [-1,1]
            rate, array = scipy_wav.read(wav)
        else:
            array, rate = soundfile.read(wav, dtype=self.dtype)
        return rate, array

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


def get_commandline_args():
    extra_chars = [' ', ';', '&', '(', ')', '|', '^', '<', '>', '?', '*',
                   '[', ']', '$', '`', '"', '\\', '!', '{', '}']

    # Escape the extra characters for shell
    argv = [arg.replace('\'', '\'\\\'\'')
            if all(char not in arg for char in extra_chars)
            else '\'' + arg.replace('\'', '\'\\\'\'') + '\''
            for arg in sys.argv]
    return sys.executable + ' ' + ' '.join(argv)


def to_recursively(vs, device, non_blocking=False):
    if isinstance(vs, dict):
        return {k: to_recursively(v, device, non_blocking)
                for k, v in value.items()}
    elif isinstance(vs, (list, tuple)):
        return type(vs)(to_recursively(v, device, non_blocking) for v in vs)
    else:
        return vs.to(device, non_blocking=non_blocking) \
            if isinstance(vs, (torch.Tensor, ComplexTensor)) else vs


pesq_url = 'https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200102-I!!SOFT-ZST-E&type=items'


def calc_pesq(ref: np.ndarray, enh: np.ndarray, fs: int) -> float:
    """Evaluate PESQ

    PESQ program can be downloaded from here:
        https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200102-I!!SOFT-ZST-E&type=items

    Reference:
        Perceptual evaluation of speech quality (PESQ)-a new method
            for speech quality assessment of telephone networks and codecs
        https://ieeexplore.ieee.org/document/941023

    Args:
        ref (np.ndarray): Reference (Nframe, Nmic)
        enh (np.ndarray): Enhanced (Nframe, Nmic)
        fs (int): Sample frequency
    """
    if shutil.which('PESQ') is None:
        raise RuntimeError(
            f"Please download from '{pesq_url}' and compile it as PESQ")
    if fs not in (8000, 16000):
        raise ValueError(f'Sample frequency must be 8000 or 16000: {fs}')
    if ref.shape != enh.shape:
        raise ValueError(f'ref and enh should have the same shape: '
                         f'{ref.shape} != {enh.shape}')
    if ref.ndim == 1:
        ref = ref[:, None]
        enh = enh[:, None]

    n_mic = ref.shape[1]
    with tempfile.TemporaryDirectory() as d:
        refs = []
        enhs = []
        for imic in range(n_mic):
            wv = str(Path(d) / f'ref.{imic}.wav')
            scipy_wav.write(wv, fs, ref[:, imic].astype(np.int16))
            refs.append(wv)

            wv = str(Path(d) / f'enh.{imic}.wav')
            scipy_wav.write(wv, fs, enh[:, imic].astype(np.int16))
            enhs.append(wv)

        lis = []
        for imic in range(n_mic):
            # PESQ +<8000|16000> <ref.wav> <enh.wav> [smos] [cond]
            commands = ['PESQ', '+{}'.format(fs),
                        refs[imic], enhs[imic]]
            with subprocess.Popen(
                    commands, stdout=subprocess.DEVNULL, cwd=d) as p:
                _, _ = p.communicate()

            # _pesq_results.txt: e.g.
            #   DEGRADED	 PESQMOS	 SUBJMOS	 COND	 SAMPLE_FREQ	 CRUDE_DELAY
            #   enh.0.wav	 2.219	 0.000	 0	 16000	-0.0080
            result_txt = (Path(d) / '_pesq_results.txt')
            if result_txt.exists():
                with result_txt.open('r') as f:
                    lis.append(float(f.readlines()[1].split()[1]))
            else:
                # Sometimes PESQ is failed. I don't know why.
                lis.append(1.)
        return float(np.mean(lis))
