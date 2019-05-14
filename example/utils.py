import collections
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.io.wavfile as scipy_wav
import soundfile
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
