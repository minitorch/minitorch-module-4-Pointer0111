from typing import Tuple, TypeVar, Any

import numpy as np
from numba import cuda
import numba

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")

# CUDA kernel for 1D convolution
@cuda.jit
def _cuda_conv1d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution CUDA kernel."""
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    # Thread indexing
    b = cuda.blockIdx.x
    oc = cuda.blockIdx.y
    x = cuda.threadIdx.x + cuda.blockIdx.z * cuda.blockDim.x

    if b >= batch_ or oc >= out_channels or x >= out_width:
        return

    s1 = input_strides
    s2 = weight_strides
    s10, s11, s12 = s1[0], s1[1], s1[2]
    s20, s21, s22 = s2[0], s2[1], s2[2]

    acc = 0.0
    w_base = oc * s20
    
    for ic in range(in_channels):
        w_ic_base = w_base + ic * s21
        in_ic_base = b * s10 + ic * s11
        
        if not reverse:
            # Anchor left: input index = x + k
            for k in range(kw):
                in_x = x + k
                if 0 <= in_x < width:
                    acc += input[in_ic_base + in_x * s12] * weight[w_ic_base + k * s22]
        else:
            # Anchor right: input index = x - k
            for k in range(kw):
                in_x = x - k
                if 0 <= in_x < width:
                    acc += input[in_ic_base + in_x * s12] * weight[w_ic_base + k * s22]
    
    out[b * out_strides[0] + oc * out_strides[1] + x * out_strides[2]] = acc


def tensor_conv1d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """CUDA wrapper for 1D convolution."""
    batch_, out_channels, out_width = out_shape
    
    # Configure grid and block dimensions
    threads_per_block = 256
    blocks_per_grid_x = batch_
    blocks_per_grid_y = out_channels
    blocks_per_grid_z = (out_width + threads_per_block - 1) // threads_per_block
    
    _cuda_conv1d_kernel[(blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z), threads_per_block](
        out, out_shape, out_strides,
        input, input_shape, input_strides,
        weight, weight_shape, weight_strides,
        reverse
    )


# CUDA kernel for 2D convolution
@cuda.jit
def _cuda_conv2d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution CUDA kernel."""
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    # Thread indexing
    b = cuda.blockIdx.x
    oc = cuda.blockIdx.y
    idx = cuda.threadIdx.x + cuda.blockIdx.z * cuda.blockDim.x
    
    if b >= batch_ or oc >= out_channels or idx >= out_height * out_width:
        return
    
    y = idx // out_width
    x = idx % out_width

    s1 = input_strides
    s2 = weight_strides
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    acc = 0.0
    w_base = oc * s20
    
    for ic in range(in_channels):
        w_ic_base = w_base + ic * s21
        in_ic_base = b * s10 + ic * s11
        
        if not reverse:
            # Top-left anchor: (y + ky, x + kx)
            for ky in range(kh):
                in_y = y + ky
                if 0 <= in_y < height:
                    wy_base = w_ic_base + ky * s22
                    for kx in range(kw):
                        in_x = x + kx
                        if 0 <= in_x < width:
                            acc += input[in_ic_base + in_y * s12 + in_x * s13] * \
                                   weight[wy_base + kx * s23]
        else:
            # Bottom-right anchor: (y - ky, x - kx)
            for ky in range(kh):
                in_y = y - ky
                if 0 <= in_y < height:
                    wy_base = w_ic_base + ky * s22
                    for kx in range(kw):
                        in_x = x - kx
                        if 0 <= in_x < width:
                            acc += input[in_ic_base + in_y * s12 + in_x * s13] * \
                                   weight[wy_base + kx * s23]
    
    out[b * out_strides[0] + oc * out_strides[1] + y * out_strides[2] + x * out_strides[3]] = acc


def tensor_conv2d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """CUDA wrapper for 2D convolution."""
    batch_, out_channels, out_height, out_width = out_shape
    
    # Configure grid and block dimensions
    threads_per_block = 256
    blocks_per_grid_x = batch_
    blocks_per_grid_y = out_channels
    total_output_size = out_height * out_width
    blocks_per_grid_z = (total_output_size + threads_per_block - 1) // threads_per_block
    
    _cuda_conv2d_kernel[(blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z), threads_per_block](
        out, out_shape, out_strides,
        input, input_shape, input_strides,
        weight, weight_shape, weight_strides,
        reverse
    )


class CudaConv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution using CUDA."""
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d_cuda(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d_cuda(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d_cuda(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


class CudaConv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution using CUDA."""
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d_cuda(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d_cuda(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d_cuda(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


cuda_conv1d = CudaConv1dFun.apply
cuda_conv2d = CudaConv2dFun.apply
