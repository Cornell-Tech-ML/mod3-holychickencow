from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function"""
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

            # Initialize output with the starting value
            out = a.zeros(tuple(out_shape))
            out_storage, _, _ = out.tuple()
            for idx in prange(len(out_storage)):
                out_storage[idx] = start

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
        # Ensure both tensors are 3-dimensional
        added_dims = 0
        if a.dimensions == 2:
            a = a.unsqueeze(0)  # Add a batch dimension
            added_dims += 1
        if b.dimensions == 2:
            b = b.unsqueeze(0)  # Add a batch dimension
            added_dims += 1
        both_2d = added_dims == 2

        # Broadcast batch dimensions
        batch_shape = shape_broadcast(a.shape[:-2], b.shape[:-2])
        final_shape = list(batch_shape) + [a.shape[-2], b.shape[-1]]
        assert a.shape[-1] == b.shape[-2], "Inner dimensions must match for matrix multiplication."

        out = a.zeros(tuple(final_shape))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Remove added batch dimension if necessary
        if both_2d:
            out = out.squeeze(0)
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
        # Fast path: aligned strides and shapes
        if (
            len(out_strides) == len(in_strides)
            and np.all(out_strides == in_strides)
            and np.all(out_shape == in_shape)
        ):
            for i in prange(out.size):
                out[i] = fn(in_storage[i])
            return

        # Compute total number of elements
        total = 1
        for dim in out_shape:
            total *= dim

        for i in prange(total):
            # Compute multi-dimensional index
            out_idx = np.empty(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_idx)

            # Broadcast index to input
            in_idx = np.empty(len(in_shape), dtype=np.int32)
            broadcast_index(out_idx, out_shape, in_shape, in_idx)

            # Compute linear positions
            out_pos = index_to_position(out_idx, out_strides)
            in_pos = index_to_position(in_idx, in_strides)

            # Apply the function
            out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)


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
        # Fast path: aligned strides and shapes
        if (
            len(out_strides) == len(a_strides) == len(b_strides)
            and np.all(out_strides == a_strides)
            and np.all(out_strides == b_strides)
            and np.all(out_shape == a_shape)
            and np.all(out_shape == b_shape)
        ):
            for i in prange(out.size):
                out[i] = fn(a_storage[i], b_storage[i])
            return

        # Compute total number of elements
        total = 1
        for dim in out_shape:
            total *= dim

        for i in prange(total):
            # Compute multi-dimensional index
            out_idx = np.empty(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_idx)

            # Broadcast index to input tensors
            a_idx = np.empty(len(a_shape), dtype=np.int32)
            b_idx = np.empty(len(b_shape), dtype=np.int32)
            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)

            # Compute linear positions
            out_pos = index_to_position(out_idx, out_strides)
            a_pos = index_to_position(a_idx, a_strides)
            b_pos = index_to_position(b_idx, b_strides)

            # Apply the zip function
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)


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
        # Compute total number of elements in output
        total = 1
        for dim in out_shape:
            total *= dim

        for i in prange(total):
            # Compute multi-dimensional index for output
            out_idx = np.empty(len(out_shape), dtype=np.int32)
            to_index(i, out_shape, out_idx)

            # Compute linear position for output
            out_pos = index_to_position(out_idx, out_strides)

            # Initialize accumulator
            acc = 0.0

            # Iterate over the reduction dimension
            for r in range(a_shape[reduce_dim]):
                a_idx = np.copy(out_idx)
                a_idx[reduce_dim] = r
                a_pos = index_to_position(a_idx, a_strides)
                acc = fn(acc, a_storage[a_pos])

            # Assign the reduced value
            out[out_pos] = acc

    return njit(_reduce, parallel=True)


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
    # Calculate the batch size
    batch_dims = len(out_shape) - 2
    batch_size = 1
    for i in range(batch_dims):
        batch_size *= out_shape[i]

    # Dimensions for matrix multiplication
    M = out_shape[-2]
    N = out_shape[-1]
    K = a_shape[-1]

    for b in prange(batch_size):
        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K):
                    a_pos = b * a_strides[0] + i * a_strides[1] + k * a_strides[2]
                    b_pos = b * b_strides[0] + k * b_strides[1] + j * b_strides[2]
                    acc += a_storage[a_pos] * b_storage[b_pos]
                out[b * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
