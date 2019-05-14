import logging
import random
import time
from pathlib import Path
from typing import List, Union, Sequence, Optional, Dict
from typing import Tuple

import humanfriendly
import numpy
import sacred
from sacred.observers import FileStorageObserver
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch_complex.functional as FC
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types
from typeguard import typechecked

from pytorch_wpe.dnn_wpe import DNN_WPE

# From local directory
from data_parallel import MyDataParallel
from utils import get_commandline_args
from utils import SoundScpReader
from utils import Stft


def get_logger(
        name='',
        format_str=
        '%(asctime)s\t[%(levelname)s]%(module)s:%(lineno)s\t%(message)s',
        date_format="%Y-%m-%d %H:%M:%S"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)

    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


global_logger = get_logger(__file__)


class Reporter:
    def __init__(self, logger: logging.Logger=global_logger):
        self.stats = {}
        self.logger = logger

    def __setitem__(self, key, value):
        self.stats.setdefault(key, []).append(value)

    def report(self, prefix: str='', nhistory: int=0):
        message = []
        for key, value in self.stats.items():
            mean = numpy.ma.masked_invalid(value[-nhistory:]).mean()
            # If there are no valid elements:
            if mean == 'masked':
                mean = 'nan'
            message.append(f'{key}={mean}')
        self.logger.info(f'{prefix}' + ' '.join(message))


@typechecked
def collate_fn(data: List[Tuple[numpy.ndarray, numpy.ndarray]]) \
        -> Tuple[ComplexTensor, ComplexTensor, torch.LongTensor]:

    # Check shape:
    for x, t in data:
        # Expected: x: (C, T, F), t: (T, F)
        assert x.ndim == 3, x.shape
        assert t.ndim == 2, t.shape
        assert x.shape[1:] == t.shape, (x.shape, t.shape)

    # Create input lengths
    ilens = numpy.array([x.shape[-2] for x, t in data], dtype=numpy.long)
    ilens = torch.from_numpy(ilens)

    # Sort by the input length
    ilens, index = torch.sort(ilens, descending=True)
    data = [data[i] for i in index]

    # From numpy to torch Tensor:
    # x: (C, T, F) -> (T, C, F)
    xs_real = [torch.from_numpy(x.real).transpose(0, 1) for x, t in data]
    xs_imag = [torch.from_numpy(x.imag).transpose(0, 1) for x, t in data]
    # t: (T, F)
    ts_real = [torch.from_numpy(t.real) for x, t in data]
    ts_imag = [torch.from_numpy(t.imag) for x, t in data]

    # Zero padding
    # xs: B x (T, C, F) -> (B, T, C, F) -> (B, C, T, F)
    xs_real = pad_sequence(xs_real, batch_first=True).transpose(1, 2)
    xs_imag = pad_sequence(xs_imag, batch_first=True).transpose(1, 2)
    # ts: B x (T, F) -> (B, T, F)
    ts_real = pad_sequence(ts_real, batch_first=True)
    ts_imag = pad_sequence(ts_imag, batch_first=True)

    xs = ComplexTensor(xs_real, xs_imag)
    ts = ComplexTensor(ts_real, ts_imag)

    # xs: (B, C, T, F), ts: (B, T, F), ilens: (B,)
    return xs, ts, ilens


class CollateFuncWithDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, data: List[Tuple[numpy.ndarray, numpy.ndarray]]) \
            -> Tuple[ComplexTensor, ComplexTensor, torch.LongTensor]:
        xs, ts, ilens = collate_fn(data)
        return xs.to(self.device), ts.to(self.device), ilens.to(self.device)


class Dataset:
    def __init__(self, x_scp: str, t_scp: str, stft_conf: str = None):
        self.x_reader = SoundScpReader(x_scp)
        self.t_reader = SoundScpReader(t_scp)
        self.keys = list(self.x_reader.keys())
        self.stft_func = Stft(stft_conf)

    def __len__(self):
        return len(self.x_reader)

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        key = self.keys[idx]

        x_rate, x = self.x_reader[key]
        # x: (C, T)
        x = x.transpose(1, 0)
        # t: (T,)
        t_rate, t = self.t_reader[key]
        assert x_rate == t_rate, (x_rate, t_rate)

        x = x.astype(numpy.double)
        t = t.astype(numpy.double)
        x = t.max() / x.max() * x

        # Stft in double precision and cast to float
        # x: (C, T) -> x_stft: (C, F, T)
        x_stft = self.stft_func(x).astype(numpy.complex64)
        # t: (T,) -> t_stft: (F, T)
        t_stft = self.stft_func(t).astype(numpy.complex64)

        # Note: Reverberant data is longer than than the original signal.
        x_stft = x_stft[..., :t_stft.shape[-1]]

        # x_stft: (C, F, T) -> (C, T, F)
        x_stft = x_stft.transpose(0, 2, 1)
        # t_stft: (F, T) -> (T, F)
        t_stft = t_stft.transpose(1, 0)

        return x_stft, t_stft


def nmse_loss(input, target):
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


class LossCalculator(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xs: ComplexTensor, ts: ComplexTensor,
                ilens: torch.LongTensor,
                loss_types: Union[str, Sequence[str]],
                ref_channel: int=0) -> Dict[str, torch.Tensor]:
        # xs: (B, C, T, F), ts: (B, T, F)
        if isinstance(loss_types, str):
            loss_types = [loss_types]

        # ys: (B, C, T, F), power: (B, C, T, F)
        ys, power = self.model(xs, ilens)
        # (B, C, T, F) -> (B, T, F)
        ys = ys[:, ref_channel]
        power = power[:, ref_channel]

        # Unpad: (B, T, F) -> (BT, F)
        uys = FC.cat(unpad(ys, ilens, length_dim=1), dim=0)
        upower = torch.cat(unpad(power, ilens, length_dim=1), dim=0)
        uts = FC.cat(unpad(ts, ilens, length_dim=1), dim=0)

        loss_dict = {}
        for loss_type in loss_types:
            if loss_type == 'wpe_nmse':
                # _ys = torch.cat([uys.real, uys.imag], dim=-1)
                # _ts = torch.cat([uts.real, uts.imag], dim=-1)
                _ys = uys.real ** 2 + uys.imag ** 2
                _ts = uts.real ** 2 + uts.imag ** 2
                # ys: (B, C, T, F) -> ys_mono: (B, T, F)
                loss = nmse_loss(_ys, _ts)

            elif loss_type == 'power_nmse':
                # ys: (B, C, T, F) -> ys_mono: (B, T, F)
                _ts = uts.real ** 2 + uts.imag ** 2
                loss = nmse_loss(upower, _ts)

            else:
                raise NotImplementedError(f'loss_type={loss_type}')

            loss_dict[loss_type] = loss

        return loss_dict


def check_gradient(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if not torch.all(torch.isfinite(param.grad)):
            return False
    return True


def train_1epoch(loss_calculator: torch.nn.Module,
                 data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 epoch: int,
                 report_interval: int=1000,
                 loss_types: Union[str, Sequence[str]]='nmse',
                 ref_channel: int=0):
    assert check_argument_types()
    reporter = Reporter()

    loss_calculator.train()
    miss_count = 0
    for idx, (xs, ts, ilens) in enumerate(data_loader):
        # xs: (B, C, T, F), ilens: (B,) -> ys: (B, C, T, F)
        optimizer.zero_grad()
        try:
            loss_dict = loss_calculator(xs, ts, ilens,
                                        loss_types=loss_types,
                                        ref_channel=ref_channel)
        except RuntimeError as e:
            # If inverse() failed in wpe
            if str(e).startswith('inverse_cuda: For batch'):
                global_logger.warning('Skipping this step. ' + str(e))
                miss_count += 1
                continue
            raise

        for loss_type, loss in loss_dict.items():
            # Averaging between each gpu devices
            loss = loss.mean()
            loss.backward()
            reporter[loss_type] = loss.item()

        if check_gradient(loss_calculator):
            optimizer.step()
        else:
            global_logger.warning('The gradient is diverged. Skip updating.')

        if (idx + 1) % report_interval == 0:
            reporter.report(f'Train {epoch}epoch '
                            f'{idx + 1}/{len(data_loader)}: ',
                            nhistory=report_interval - miss_count)
            miss_count = 0


def test(loss_calculator: torch.nn.Module,
         data_loader: DataLoader,
         epoch: int,
         loss_types: Union[str, Sequence[str]]='nmse',
         ref_channel: int=0):
    assert check_argument_types()
    reporter = Reporter()

    loss_calculator.eval()
    for xs, ts, ilens in data_loader:
        with torch.no_grad():
            loss_dict = loss_calculator(xs, ts, ilens,
                                        loss_types=loss_types,
                                        ref_channel=ref_channel)
            for loss_type, loss in loss_dict.items():
                # Averaging between each gpu devices
                loss = loss.mean()
                reporter[loss_type] = loss.item()
    reporter.report(f'Eval {epoch}epoch: {len(data_loader)}sample: ')


ex = sacred.Experiment('train')
ex.logger = global_logger


@ex.config
def config():
    """Configuration for training

    This script depends on sacred for CLI system.
    You can change the configuration: e.g.

        % python train.py with batch_size=32 workdir=my_exp
        % python train.py with opt_config.lr=0.1
    """
    # Output directory
    workdir = 'exp'

    # Input files
    train_x_scp = 'data/train/reverb.scp'
    train_t_scp = 'data/train/clean.scp'
    test_x_scp = 'data/dev/reverb.scp'
    test_t_scp = 'data/dev/clean.scp'
    stft_conf = './stft.json'

    seed = 0
    # None indicates using all visible devices
    ngpu = None
    batch_size = 1
    nepoch = 100
    report_interval = 100
    resume = None

    # 'SGD', 'Adam'
    opt_type = 'Adam'
    opt_config = {'lr': 0.001}

    loss_type = 'wpe_nmse'
    # The reference channel used for loss calculation
    ref_channel = 0

    Path(workdir).mkdir(parents=True, exist_ok=True)
    observer = FileStorageObserver.create(str(Path(workdir) / 'config'))
    ex.observers.append(observer)
    del observer


@ex.automain
def main(seed: int,
         ngpu: Optional[int],
         batch_size: int,
         workdir: str,
         train_x_scp: str,
         train_t_scp: str,
         test_x_scp: str,
         test_t_scp: str,
         stft_conf: str,
         opt_type: str,
         opt_config: dict,
         report_interval: int,
         nepoch: int,
         loss_type: str,
         ref_channel: int):
    global_logger.info(get_commandline_args())
    assert check_argument_types()
    if ngpu is None:
        ngpu = torch.cuda.device_count()

    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    _collate_fn = CollateFuncWithDevice('cuda' if ngpu > 1 else 'cpu')
    train_loader = DataLoader(
        dataset=Dataset(train_x_scp, train_t_scp, stft_conf),
        batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

    test_loader = DataLoader(
        dataset=Dataset(test_x_scp, test_t_scp, stft_conf),
        batch_size=1, shuffle=False, collate_fn=_collate_fn)

    # MyDataParallel can handle "ComplexTensor"
    model = DNN_WPE()
    loss_calculator = MyDataParallel(
        LossCalculator(model), device_ids=list(range(ngpu)))

    OptimzerClass = getattr(torch.optim, opt_type)
    optimizer = OptimzerClass(model.parameters(), **opt_config)
    global_logger.info(model)
    global_logger.info(optimizer)

    elapsed = {}
    for epoch in range(1, nepoch + 1):
        global_logger.info(f'Start {epoch}/{nepoch} epoch')
        t = time.perf_counter()

        train_1epoch(loss_calculator, train_loader, optimizer,
                     epoch=epoch, report_interval=report_interval,
                     loss_types=loss_type, ref_channel=ref_channel)

        test(loss_calculator, test_loader, epoch=epoch,
             loss_types=['wpe_nmse',
                         'power_nmse'],
             ref_channel=ref_channel)

        torch.save(model.state_dict(), f'{workdir}/model_{epoch}.pt')
        torch.save(optimizer.state_dict(), f'{workdir}/optimizer_{epoch}.pt')

        elapsed[epoch] = time.perf_counter() - t
        this = humanfriendly.format_timespan(elapsed[epoch])
        mean = numpy.mean(list(elapsed.values()))
        expect = humanfriendly.format_timespan(mean * (nepoch - epoch))
        global_logger.info(f'Elapsed time for {epoch}epoch: {this}, '
                           f'Expected remaining time: {expect}')
