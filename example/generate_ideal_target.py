import argparse
import random
from pathlib import Path
from typing import List

import h5py
import numpy
import torch
from torch.utils.data import DataLoader
from torch_complex.tensor import ComplexTensor
from tqdm import tqdm
from typeguard import check_argument_types

from pytorch_wpe import wpe_one_iteration

# From local directory
from dataset import WavRIRNoiseDataset, collate_fn


def nmse_loss(input, target):
    """Normalized Mean Squared Error"""
    assert check_argument_types()
    assert input.shape == target.shape, (input.shape, target.shape)

    loss = torch.norm(input - target) ** 2 / torch.norm(target) ** 2
    return loss


def unpad(x: torch.Tensor, ilens: torch.LongTensor,
          length_dim: int=1) -> List[torch.Tensor]:
    # x: (B, ..., T, ...)
    if length_dim < 0:
        length_dim = x.dim() + length_dim

    xs = []
    for _x, l in zip(x, ilens):
        # ind = (:, ..., :l, , :)
        ind = tuple(slice(0, l) if i == length_dim else slice(None)
                    for i in range(1, x.dim()))
        # _x: (T, ...) -> (l, ...)
        xs.append(_x[ind])
    # B x (..., li, ...)
    return xs


def check_gradient(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if not torch.all(torch.isfinite(param.grad)):
            return False
    return True


def optimize_input(x: ComplexTensor, t: ComplexTensor,
                   l: torch.LongTensor,
                   niter=10, taps=5, delay=3) -> torch.Tensor:
    def loss_func(x, power, t_cat):
        # x: (F, C, T), power: (T, F) -> enhanced: (F, C, T)
        enhanced = wpe_one_iteration(x, power.transpose(0, 1),
                                     taps=taps, delay=delay)
        # enhanced_cat: (2 * F, C, T)
        enhanced_cat = torch.cat([enhanced.real, enhanced.imag], dim=0)
        # t_cat: (C, T, 2 * F) -> (2 * F, C, T)
        t_cat = t_cat.permute(2, 0, 1)
        return enhanced, nmse_loss(enhanced_cat, t_cat)

    # x: (C, T, F)
    x = x[:, :l]
    # t: (C, T, F)
    t = t[:, :l]
    # mean_power: (T, F)
    init_power = (x.real ** 2 + x.imag ** 2).mean(dim=0)
    # mask: (T, F)
    mask = torch.nn.Parameter(torch.full_like(init_power, 10.))
    opt = torch.optim.SGD([mask], lr=1.)

    # x: (C, T, F) -> (F, C, T)
    x = x.permute(2, 0, 1).contiguous()
    t_cat = torch.cat([t.real, t.imag], dim=-1)

    best_loss = None
    best_power = None
    for _ in range(niter):
        opt.zero_grad()

        # power: (T, F)
        power = init_power * torch.sigmoid(mask)
        _, loss = loss_func(x, power, t_cat)

        if best_loss is None or loss.item() < best_loss:
            best_loss = loss.item()
            best_power = power.clone().detach()

        loss.backward()
        opt.step()

    # wpe with x
    with torch.no_grad():
        # e: (F, C, T)
        e, init_loss = loss_func(x, init_power, t_cat)
        wpe_power = (e.real ** 2 + e.imag ** 2).mean(dim=1).transpose(0, 1)
        _, wpe_loss = loss_func(x, wpe_power, t_cat)

    # wpe with t
    with torch.no_grad():
        # t: (C, T, F) -> (T, F)
        t_power = (t.real ** 2 + t.imag ** 2).mean(dim=0)
        # e: (F, C, T)
        e, t_loss = loss_func(x, t_power, t_cat)
        t_power2 = (e.real ** 2 + e.imag ** 2).mean(dim=1).transpose(0, 1)
        _, t_loss2 = loss_func(x, t_power2, t_cat)

    pairs = (['input', init_loss.item(), init_power],
             ['wpe with input', wpe_loss.item(), wpe_power],
             ['target', t_loss.item(), t_power],
             ['wpe with target', t_loss2.item(), t_power2],
             ['estimate', best_loss, best_power])

    idx = int(numpy.argmin([p[1] for p in pairs]))
    return pairs[idx][2], {'what': pairs[idx][0],
                           'gain': -(pairs[idx][1] - pairs[0][1])}


def main(seed: int,
         in_scp: str,
         out_h5: str,
         out_info: str,
         stft_conf: str,
         nworker: int):
    assert check_argument_types()
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    try:
        with h5py.File(out_h5, 'w') as f, open(out_info, 'w') as fwhat:
            _train_loader = DataLoader(
                dataset=WavRIRNoiseDataset(in_scp, stft_conf=stft_conf),
                collate_fn=collate_fn, num_workers=nworker)
            for keys, xs, ts, _, ilens in tqdm(_train_loader):
                for k, x, t, l in zip(keys, xs, ts, ilens):
                    power, info = optimize_input(x, t, l)
                    f.create_dataset(k, data=power.cpu().numpy(),
                                     compression='gzip')
                    fwhat.write(f'{k} {info}\n')
    except:
        if Path(out_h5).exists():
            Path(out_h5).unlink()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-scp', type=str, required=True)
    parser.add_argument('--out-h5', type=str, required=True)
    parser.add_argument('--out-info', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stft_conf', type=str, default='./stft.json')
    parser.add_argument('--nworker', type=int, default=1)
    args = parser.parse_args()

    main(seed=args.seed,
         in_scp=args.in_scp,
         out_h5=args.out_h5,
         out_info=args.out_info,
         stft_conf=args.stft_conf,
         nworker=args.nworker)
