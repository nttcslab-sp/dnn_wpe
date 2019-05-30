import argparse
from typing import Tuple

import torch
import numpy as np

from example.model import DNN_WPE

from data_parallel import MyDataParallel
from utils import SoundScpReader
from utils import SoundScpWriter
from utils import Stft


def str2int_tuple(st: str) -> Tuple[int]:
    sps = st.split(',')
    return tuple(int(s) for s in sps)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-scp', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--stft-file', type=str, default='./stft.json')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--ref-channel', type=str2int_tuple, default=None)

    args = parser.parse_args()

    model = MyDataParallel(
        DNN_WPE(), device_ids=list(range(args.ngpu)))

    state = torch.load(args.model_file)
    model.load_state_dict(state)

    reader = SoundScpReader(args.in_scp)
    writer = SoundScpWriter(args.out_dir, 'wav')
    stft_func = Stft(args.stft_file)

    for key in reader:
        # inp: (T, C)
        rate, inp = reader[key]
        # Scaling int to [-1, 1]
        inp = inp.astype(np.float32) / (np.iinfo(inp.dtype).max - 1)

        # inp: (T, C) -> inp_stft: (C, F, T)
        inp_stft = stft_func(inp.T)
        # (C, F, T) -> (B, C, T, F)
        inp_stft = inp_stft.transpose(0, 2, 1)[None]
        # To torch
        inp_stft_th = torch.from_numpy(inp_stft).to(
            'cuda' if args.ngpu > 1 else 'cpu')

        # Apply DNN enh_stft_th: (B, C, T, F)
        enh_stft_th, _ = model(inp_stft_th, return_wpe=True)
        # enh_stft_th: (B, C, T, F) -> (C, F, T)
        enh_stft_th = enh_stft_th[0].permute(0, 2, 1)

        # To numpy
        enh_stft = enh_stft_th.cpu().numpy()
        # enh_stft: (C, F, T) -> enh: (T, C)
        enh = stft_func.istft(enh_stft).T
        # Truncate
        enh = enh[:inp.shape[0]]
        # Rescaling  [-1, 1] to int16
        enh = (enh * (np.iinfo(np.int16).max - 1)).astype(np.int16)

        if args.ref_channels is not None:
            enh = enh[:, args.ref_channels]
        writer[key] = enh


if __name__ == '__main__':
    main()

