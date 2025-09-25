from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # 中文说明：
        # - 并行：使用 prange 对外层元素循环进行并行切分（numba 会在多线程间分配 i 的区间）。
        # - 索引缓冲：所有多维索引缓冲使用 numpy 数组，避免 Python 对象参与循环。
        # - 对齐优化：当 out 与 in 在形状和 strides 完全一致时，跳过广播与索引计算，直接同一位置读写。
        n = len(out)
        same_shape = len(out_shape) == len(in_shape)
        if same_shape:
            for d in range(len(out_shape)):
                if out_shape[d] != in_shape[d]:
                    same_shape = False
                    break
        aligned = same_shape
        if aligned:
            for d in range(len(out_strides)):
                if out_strides[d] != in_strides[d]:
                    aligned = False
                    break
        for i in prange(n):
            if aligned:
                # 对齐路径：只需计算一次 out 的线性位置，直接用相同位置访问 in/out
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] = fn(in_storage[out_pos])
            else:
                # 非对齐/需广播路径：根据 out_index 计算 in_index，再各自转位置
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                in_index = np.zeros(len(in_shape), dtype=np.int32)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                in_pos = index_to_position(in_index, in_strides)
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # 中文说明：
        # - 并行：对 out 的每个元素位置并行计算 a 与 b 对应（含广播）的值，应用二元 fn。
        # - 对齐优化：当 out/a/b 形状与 strides 完全一致时，直接用相同线性位置读写，省去三次索引转换。
        n = len(out)
        same_shape = (
            len(out_shape) == len(a_shape) == len(b_shape)
        )
        if same_shape:
            for d in range(len(out_shape)):
                if not (out_shape[d] == a_shape[d] == b_shape[d]):
                    same_shape = False
                    break
        aligned = same_shape
        if aligned:
            for d in range(len(out_strides)):
                if not (
                    out_strides[d] == a_strides[d] == b_strides[d]
                ):
                    aligned = False
                    break
        for i in prange(n):
            if aligned:
                # 对齐路径：只算一次 out 的位置，同步用于 a、b、out
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                pos = index_to_position(out_index, out_strides)
                out[pos] = fn(a_storage[pos], b_storage[pos])
            else:
                # 非对齐/广播路径：为 a、b 分别计算广播后的索引与线性位置
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                a_index = np.zeros(len(a_shape), dtype=np.int32)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                b_index = np.zeros(len(b_shape), dtype=np.int32)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # 中文说明：
        # - 并行：对每个 out 位置（即已压缩维为 1 的形状）并行做一条独立的归约。
        # - 内层循环优化：先计算 a 的起始线性位置与步长 step，仅做简单的线性步进和 fn 调用，无额外索引函数调用。
        n = len(out)
        reduce_size = a_shape[reduce_dim]
        for i in prange(n):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            acc = out[out_pos]

            # 构造在 reduce_dim = 0 时的基准索引，然后通过步长线性前进
            a_index = out_index.copy()
            a_index[reduce_dim] = 0
            a_pos = index_to_position(a_index, a_strides)
            step = a_strides[reduce_dim]

            # 仅进行内存线性访问与函数累积，避免在内层做昂贵的索引变换
            for j in range(reduce_size):
                acc = fn(acc, a_storage[a_pos + j * step])

            out[out_pos] = acc

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # 中文说明：外层并行遍历 out 的所有 (n, i, j) 位置；
    # 使用 strides 做线性地址计算；内层仅做一次乘法与本地累加，无全局写。
    N = out_shape[0]
    I = out_shape[1]
    J = out_shape[2]
    K = a_shape[2]  # a_shape[-1] == b_shape[-2]

    total = N * I * J
    for idx in prange(total):
        # 反算 (n, i, j)，避免调用任何索引函数
        n = idx // (I * J)
        rem = idx - n * (I * J)
        i = rem // J
        j = rem - i * J

        # 线性位置（out）
        out_pos = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]

        # 批次广播：当某输入的 batch 维为 1 时，其 batch stride 设为 0
        a_batch_off = n * a_batch_stride
        b_batch_off = n * b_batch_stride

        # 累加到局部寄存器
        acc = 0.0
        a_row_base = a_batch_off + i * a_strides[1]
        b_col_base = b_batch_off + j * b_strides[2]
        a_k_stride = a_strides[2]
        b_k_stride = b_strides[1]

        for k in range(K):
            acc += a_storage[a_row_base + k * a_k_stride] * b_storage[b_col_base + k * b_k_stride]

        # 单次全局写
        out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
