import torch
from torch.nn.parallel._functions import Gather
from torch.nn.parallel._functions import Scatter
from torch_complex.tensor import ComplexTensor


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, ComplexTensor):
            sreal = Scatter.apply(target_gpus, None, dim, obj.real)
            simag = Scatter.apply(target_gpus, None, dim, obj.imag)
            return tuple(ComplexTensor(r, i) for r, i in zip(sreal, simag))
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if isinstance(out, ComplexTensor):
            return ComplexTensor(
                Gather.apply(target_device, dim, *[o.real for o in outputs]),
                Gather.apply(target_device, dim, *[o.imag for o in outputs]))
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class MyDataParallel(torch.nn.DataParallel):
    """Override torch.nn.DataParallel to support
    scattering and gathering for ComplexTensor """
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids, dim=0):
        inputs = scatter(inputs, device_ids, dim) if inputs else []
        kwargs = scatter(kwargs, device_ids, dim) if kwargs else []

        # if len(inputs) < len(kwargs):
        #     inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        # elif len(kwargs) < len(inputs):
        #     kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        inputs = inputs[:len(kwargs)]
        kwargs = kwargs[:len(inputs)]
        return inputs, kwargs

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def test():
    class Net(torch.nn.Module):
        def forward(self, x):
            assert isinstance(x, ComplexTensor), type(x)
            return x

    net = MyDataParallel(Net(), device_ids=[0, 1])
    data = ComplexTensor(torch.randn(4, 100),
                         torch.randn(4, 100)).cuda()

    out = net(data)

    assert isinstance(data, ComplexTensor), type(data)
    assert torch.all(out.real == data.real)
    assert torch.all(out.imag == data.imag)


if __name__ == '__main__':
    test()
