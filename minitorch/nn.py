from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_h = height // kh
    new_w = width // kw
    # (B, C, H, W) -> (B, C, new_h, kh, new_w, kw)
    t = input.contiguous().view(batch, channel, new_h, kh, new_w, kw)
    # -> (B, C, new_h, new_w, kh, kw)
    t = t.permute(0, 1, 2, 4, 3, 5)
    # -> (B, C, new_h, new_w, kh*kw)
    t = t.contiguous().view(batch, channel, new_h, new_w, kh * kw)
    return t, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    t, new_h, new_w = tile(input, kernel)
    kh, kw = kernel
    # sum over last dim then divide by window size
    out = t.sum(dim=4).view(input.shape[0], input.shape[1], new_h, new_w) / (kh * kw)
    return out


# Max reduction along a dimension using FastOps.reduce with operators.max
_reduce_max = FastOps.reduce(operators.max, start=-1e30)


def max(input: Tensor, dim: int) -> Tensor:  # type: ignore[override]
    return _reduce_max(input, dim)


def argmax(input: Tensor, dim: int) -> Tensor:
    # one-hot along dim: eq to max broadcasted
    m = max(input, dim)
    return input == m  # type: ignore[return-value]


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    t, new_h, new_w = tile(input, kernel)
    out = max(t, 4).view(input.shape[0], input.shape[1], new_h, new_w)
    return out


def softmax(input: Tensor, dim: int) -> Tensor:
    # stable softmax: exp(x - max) / sum(exp(x - max))
    m = max(input, dim)
    shifted = input - m
    ex = shifted.exp()
    z = ex.sum(dim)
    return ex / z


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    m = max(input, dim)
    shifted = input - m
    log_z = shifted.exp().sum(dim).log()
    return shifted - log_z


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    if ignore:
        return input
    # mask_keep = 1 where r >= p, else 0
    r = rand(input.shape)
    keep = (r < p) == 0.0
    return input * keep
