# -*- coding: utf-8 -*-
'''
@File    : concat_batch_tensor.py
@Time    : 2026/05/01 23:51:43
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Implement ConcateBatchTensor to store a batch of
           variable-length (shape[0]) tensors in one contiguous tensor.
'''

import torch
from collections.abc import Sequence


class ConcatBatchTensor:
    def __init__(
        self,
        tensors: list[torch.Tensor] | None = None,
        batch_sizes: Sequence[int] | None = None,
        data_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        """Store a batch of variable-length (shape[0]) tensors in one contiguous tensor.
        If use padding to store, the shape of this tensor is [N, max(batch_sizes), *data_shape].
        But with this implementation, we will store the tensors in a concatenated way,
        and the shape of this tensor is [sum(batch_sizes), *data_shape].
        
        Initialize the ConcatBatchTensor has two ways:
        1. Provide a list of tensors to concatenate, shape[0] diff, shape[1:] same.
        2. Provide the batch sizes and single data shape to create an empty concatenated tensor.

        Example usage:
        ```python
        # Init 1
        tensors = [torch.randn(3, 2), torch.randn(2, 2), torch.randn(4, 2)]
        cbt = ConcatBatchTensor(tensors=tensors)
        
        # Init 2
        batch_sizes = [3, 2, 4]
        data_shape = (2,)
        cbt = ConcatBatchTensor(batch_sizes=batch_sizes, data_shape=data_shape)
        ```

        Args:
            tensors: a list of tensors to concatenate.
            batch_sizes: a list of batch sizes for each tensor. Only used when tensors is None.
            data_shape: the shape of the data for each tensor. Only used when tensors is None.
            dtype: the data type of the concatenated tensor. Only used when tensors is None.
            device: the device of the concatenated tensor.
        """
        self._device = torch.device(device)

        if tensors is not None:
            if len(tensors) == 0:
                raise ValueError("tensors must not be empty.")
            self._batch_sizes = torch.as_tensor(
                [tensor.shape[0] for tensor in tensors], dtype=torch.int64, device=self._device
            )
            self._data_shape = tuple(tensors[0].shape[1:])
            for i, tensor in enumerate(tensors):
                if tuple(tensor.shape[1:]) != self._data_shape:
                    raise ValueError(f"All tensors must have the same shape except the first dimension. Got {tensor.shape} at index {i}, expected shape (*, {self._data_shape}).")
            self._concatenated_tensor = torch.cat(tensors, dim=0).to(device=self._device)
        else:
            if batch_sizes is None or data_shape is None:
                raise ValueError("Either tensors or both batch_sizes and data_shape must be provided.")
            self._batch_sizes = torch.as_tensor(batch_sizes, dtype=torch.int64, device=self._device)
            self._data_shape = tuple(data_shape)
            total_size = int(self._batch_sizes.sum().item())
            self._concatenated_tensor = torch.empty((total_size, *self._data_shape), dtype=dtype, device=self._device)

        self._batch_ends = torch.cumsum(self._batch_sizes, dim=0)
        self._batch_starts = torch.empty_like(self._batch_ends)
        self._batch_starts[0] = 0
        self._batch_starts[1:] = self._batch_ends[:-1]

    def __len__(self) -> int:
        return len(self._batch_sizes)

    def gather(self, batch_idx: int | list | torch.Tensor, data_idx: int | list | torch.Tensor) -> torch.Tensor:
        """Gather one item per batch using a specialized fast path.

        Args:
            batch_idx: Batch indices to gather from.
            data_idx: In-batch indices aligned with `batch_idx`.

        Returns:
            Gathered tensor values with shape `(*batch_idx.shape, *data_shape)`.
        """
        if isinstance(batch_idx, (int, list)):
            batch_idx = torch.as_tensor(batch_idx, dtype=torch.int64, device=self._device)
        batch_idx = batch_idx.to(dtype=torch.int64, device=self._device)
        if isinstance(data_idx, (int, list)):
            data_idx = torch.as_tensor(data_idx, dtype=torch.int64, device=self._device)
        data_idx = data_idx.to(dtype=torch.int64, device=self._device)

        if (data_idx < 0).any() or (batch_idx < 0).any():
            raise IndexError(f"Negative index is not supported. Got {batch_idx=}, {data_idx=}.")
        if (data_idx >= self._batch_sizes[batch_idx]).any():
            raise IndexError("data_idx is out of range for at least one batch item.")

        flat_idx = self._batch_starts[batch_idx] + data_idx
        return self._concatenated_tensor[flat_idx]

    def __getitem__(self, idx: tuple[int | list | torch.Tensor, int | list | torch.Tensor] | int | list | torch.Tensor) -> torch.Tensor:
        """Support 6 types of indexing:
        - cbt[batch_idx]: return batch index, shape (batch_size, *data_shape).
        - cbt[batch_idxs]: return batch indices, shape (sum(batch_sizes), *data_shape).
        - cbt[batch_idx, data_idx]: return one-to-one index, shape (*data_shape).
        - cbt[batch_idxs, data_idx]: return multi-to-one index, shape (len(batch_idxs), *data_shape).
        - cbt[batch_idx, data_idxs]: return one-to-multi index, shape (len(data_idxs), *data_shape).
        - cbt[batch_idxs, data_idxs]: return one-to-one index, shape (len(batch_idxs), *data_shape).
        """
        if isinstance(idx, tuple):
            batch_idx, data_idx = idx
            if isinstance(batch_idx, (int, list)):
                batch_idx = torch.as_tensor(batch_idx, dtype=torch.int64, device=self._device)
            batch_idx = batch_idx.to(dtype=torch.int64, device=self._device)
            if isinstance(data_idx, (int, list)):
                data_idx = torch.as_tensor(data_idx, dtype=torch.int64, device=self._device)
            data_idx = data_idx.to(dtype=torch.int64, device=self._device)

            # don't support negative index
            if (data_idx < 0).any() or (batch_idx < 0).any():
                raise IndexError(f"Negative index is not supported. Got {batch_idx=}, {data_idx=}.")

            # check data_idx is in range
            if (data_idx >= self._batch_sizes[batch_idx]).any():
                if batch_idx.numel() == 1:
                    raise IndexError(f"{data_idx=} out of batch_size={self._batch_sizes[batch_idx]} at {batch_idx=}.")
                else:
                    for i, b_idx in enumerate(batch_idx):
                        if (data_idx.ndim == 1 and data_idx[i] >= self._batch_sizes[b_idx]):
                            raise IndexError(f"{data_idx[i].item()} out of batch_size={self._batch_sizes[b_idx]} at batch_index={b_idx.item()}.")
                        if (data_idx.ndim == 0 and data_idx >= self._batch_sizes[b_idx]):
                            raise IndexError(f"data_idx={data_idx.item()} out of batch_size={self._batch_sizes[b_idx]} at batch_index={b_idx.item()}.")

            return self.gather(batch_idx, data_idx)

        if isinstance(idx, int):
            # single batch index
            start = self._batch_starts[idx]
            end = self._batch_ends[idx]
            return self._concatenated_tensor[start:end]
        
        # list or tensor of batch indices
        idx = torch.as_tensor(idx, dtype=torch.int64, device=self._device)
        return torch.cat([self._concatenated_tensor[self._batch_starts[i] : self._batch_ends[i]] for i in idx], dim=0)

    def __setitem__(self, idx: int, data: torch.Tensor) -> None:
        """Set the data only support single batch index"""
        start = int(self._batch_starts[idx])
        end = int(self._batch_ends[idx])
        self._concatenated_tensor[start:end] = data

    @property
    def batch_sizes(self) -> torch.Tensor:
        return self._batch_sizes

    @property
    def data_shape(self) -> tuple:
        return self._data_shape

    @property
    def shape(self) -> tuple:
        return self._concatenated_tensor.shape


if __name__ == "__main__":
    cbt = ConcatBatchTensor(batch_sizes=[3, 2, 4], data_shape=(2,))
    cbt[0] = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    cbt[1] = torch.tensor([[4.0, 4.0], [5.0, 5.0]])
    cbt[2] = torch.tensor([[6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [9.0, 9.0]])

    print("len:", len(cbt))
    print("shape:", cbt.shape)
    print("batch 0:\n", cbt[0])
    print("batch 2:\n", cbt[2])
    print(
        "gather:\n",
        cbt[
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 0, 3]),
        ],
    )
    print(cbt[[0,1,2], [1,0,2]])
    print(cbt[[0,1]])
    print(cbt[[0,1], 1])
