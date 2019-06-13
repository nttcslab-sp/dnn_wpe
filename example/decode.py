import argparse
import json
from distutils.util import strtobool
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor
from tqdm import tqdm
import numpy as np

from model import DNN_WPE
from utils import SoundScpReader
from utils import SoundScpWriter
from utils import Stft


def str2int_tuple(st: str) -> Tuple[int]:
    if st == 'none':
        return None
    sps = st.split(',')
    return tuple(int(s) for s in sps)


def wpe(Y: np.ndarray, taps=5, delay=3, iterations=3, psd_context=0,
        statistics_mode='full',
        power: np.ndarray=None):
    import nara_wpe.wpe

    # Y: (F, C, T)
    X = Y
    Y_tilde = nara_wpe.wpe.build_y_tilde(Y, taps, delay)

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    for i in range(iterations):
        if power is None:
            inverse_power = nara_wpe.wpe.get_power_inverse(X, psd_context=psd_context)
        else:
            inverse_power = 1 / power
        G = nara_wpe.wpe.get_filter_matrix_v7(
            Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=inverse_power[s])
        X = nara_wpe.wpe.perform_filter_operation_v5(Y=Y, Y_tilde=Y_tilde,
                                                     filter_matrix=G)
    return X


def online_wpe(Y: np.ndarray, taps=5, delay=3, alpha=0.9999, power=None):
    import nara_wpe.wpe
    # Y: (F, C, T)
    channels = Y.shape[1]
    frequency_bins = Y.shape[0]
    Q = np.stack([np.identity(channels * taps) for _ in range(frequency_bins)])
    G = np.zeros((frequency_bins, channels * taps, channels))

    Z_list = [Y[:, :, :taps + delay + 1]]
    for iframe in range(Y.shape[2] - (taps + delay + 1)):
        # yframe: (F, C, taps + delay + 1)
        yframe = Y[:, :, iframe:iframe + taps + delay + 1]

        if power is None:
            _power = nara_wpe.wpe.get_power_online(
                # yframe
                Y[:, :, iframe + taps + delay:iframe + taps + delay + 2]
            )
        else:
            # power: (F, T)
            _power = power[:, iframe:iframe + taps + delay + 1].mean(1)
        Z, _Q, _G = nara_wpe.wpe.online_wpe_step(yframe.transpose(2, 0, 1),
                                                 _power, Q, G,
                                                 alpha=alpha, delay=delay,
                                                 taps=taps)
        if power is None:
            Q, G = _Q, _G
        Z_list.append(Z[:, :, None])
    return np.concatenate(Z_list, axis=2)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-scp', type=str, required=True)
    parser.add_argument('--clean-scp', type=str,
                        help='Decode using oracle clean power')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--model-state', type=str)
    parser.add_argument('--model-config', type=str)
    parser.add_argument('--stft-file', type=str, default='./stft.json')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--ref-channels', type=str2int_tuple, default=None)
    parser.add_argument('--online', type=strtobool, default=False)
    parser.add_argument('--taps', type=int, default=5)
    parser.add_argument('--delay', type=int, default=3)

    args = parser.parse_args()
    devcice = 'cuda' if args.ngpu > 1 else 'cpu'

    if args.model_config is not None:
        with open(args.model_config) as f:
            model_config = json.load(f)

        model_config.update(use_dnn=True)
        _ = model_config.pop('width')
        norm_scale = model_config.pop('norm_scale')
        model = DNN_WPE(**model_config)
        if args.model_state is not None:
            model.load_state_dict(torch.load(args.model_state))
    else:
        model = None
        norm_scale = False

    reader = SoundScpReader(args.in_scp)
    writer = SoundScpWriter(args.out_dir, 'wav')

    if args.clean_scp is not None:
        clean_reader = SoundScpReader(args.clean_scp)
    else:
        clean_reader = None
    stft_func = Stft(args.stft_file)

    for key in tqdm(reader):
        # inp: (T, C)
        rate, inp = reader[key]
        if inp.ndim == 1:
            inp = inp[:, None]
        if args.ref_channels is not None:
            inp = inp[:, args.ref_channels]

        # Scaling int to [-1, 1]
        inp = inp.astype(np.float32) / (np.iinfo(inp.dtype).max - 1)
        if norm_scale:
            scale = np.abs(inp).mean()
            inp /= scale
        else:
            scale = 1.

        # inp: (T, C) -> inp_stft: (C, F, T)
        inp_stft = stft_func(inp.T)

        if clean_reader is not None:
            _, clean = clean_reader[key]
            if clean.ndim == 1:
                clean = clean[:, None]
            # clean: (T, C) -> clean_stft: (C, F, T)
            clean_stft = stft_func(clean.T)
            power = (clean_stft.real ** clean_stft.imag ** 2).mean(0)

        elif model is not None:
            # To torch(C, F, T) -> (1, C, T, F)
            inp_stft_th = ComplexTensor(inp_stft.transpose(0, 2, 1)[None]).to(devcice)
            with torch.no_grad():
                _, power = model(inp_stft_th, return_wpe=False)
            # power: (1, C, T, F) -> (F, C, T)
            power = power[0].permute(2, 0, 1)

            # To numpy: (F, C, T) -> (F, T)
            power = power.cpu().numpy().mean(1)
        else:
            power = None

        # enh_stft: (F, C, T)
        if not args.online:
            enh_stft = wpe(inp_stft.transpose(1, 0, 2),
                           power=power, taps=args.taps, delay=args.delay,
                           iterations=1 if model is not None else 3,)
        else:
            enh_stft = online_wpe(inp_stft.transpose(1, 0, 2),
                                  power=power, taps=args.taps,
                                  delay=args.delay)
        # enh_stft: (F, C, T) -> (C, F, T)
        enh_stft = enh_stft.transpose(1, 0, 2)
        enh_stft = enh_stft[0]

        # enh_stft: (C, F, T) -> enh: (T, C)
        enh = stft_func.istft(enh_stft).T
        # Truncate
        enh = enh[:inp.shape[0]]

        if norm_scale:
            enh *= scale
        # Rescaling  [-1, 1] to int16
        enh = (enh * (np.iinfo(np.int16).max - 1)).astype(np.int16)

        writer[key] = (rate, enh)


if __name__ == '__main__':
    main()

