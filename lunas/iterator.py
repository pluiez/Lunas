"""
This module defines iterators to arrange a list of samples as a batch and process the samples to be fed into a model.
"""
from __future__ import annotations

import abc
import bisect
import collections
import itertools
import math
import warnings
from typing import *

import numpy

import lunas.dataset.core as core

__all__ = ['BatchIterator', 'ConstantIterator', 'BucketIterator', 'DataLoader', 'get_bucket_boundaries']

try:
    from torch.utils.data.dataset import IterableDataset as _IterableDataset
except ImportError:
    class _IterableDataset(object):
        ...


class BatchIterator(_IterableDataset, abc.ABC):

    def __init__(self, data_source: Union[core.Dataset, BatchIterator]) -> None:
        """Initialises the iterator.

        Args:
            data_source: an instance of Dataset or BatchIterator.
        """
        if not isinstance(data_source, (core.Dataset, BatchIterator)):
            raise TypeError(f'dataset ({type(data_source)}) must be instance of Dataset or BatchIterator.')

        self._data_source = data_source

    @abc.abstractmethod
    def generator(self) -> Iterator:
        raise NotImplementedError

    def state(self):
        return self._data_source.state()

    def load(self, state):
        self._data_source.load(state)

    def __iter__(self):
        num_workers = 1
        worker_id = 0

        try:
            import torch
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
        except ImportError:
            ...

        i = 0
        for samples in self.generator():
            if i == worker_id:
                yield samples
            i = (i + 1) % num_workers


class DataLoader(object):
    """Load and batch data.

    DataLoader wraps around a BatchIterator instance, batches multiple samples and optionally moves the data to pinned
    memory.

    This DataLoader relies on PyTorch's carefully designed `DataLoader` to implement safe multiprocessing operations.
    Since in a naive implementation, the main Python process can lose connection with child processes if one of them
    crashed, for example, by segment fault. However, PyTorch's DataLoader deals with this problem by tracking each
    process's states manually. Here we reuse it for simplicity.
    """

    def __init__(
            self,
            batch_iterator: BatchIterator,
            num_workers: int = 0,
            collate_fn: Callable[[List], Any] = None,
            pin_memory: bool = False,
            timeout: int = 0,
            worker_init_fn: Callable[[int], None] = None,
            multiprocessing_context=None
    ):
        if not isinstance(batch_iterator, BatchIterator):
            raise TypeError(f'batch_iterator ({type(batch_iterator)}) must be instance of BatchIterator.')

        self._batch_iterator = batch_iterator
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._timeout = timeout
        self._worker_init_fn = worker_init_fn
        self._multiprocessing_context = multiprocessing_context
        self._warning = False

        self._ptr = 0
        self._resumed = False

    def state(self):
        return {'ptr': self._ptr}

    def load(self, state):
        if not state:
            return
        self._ptr = state['ptr']
        self._resumed = True

    def _init_worker(self, worker_id):
        numpy.random.seed(numpy.random.get_state()[1][0])
        if self._worker_init_fn is not None:
            self._worker_init_fn(worker_id)

    def __iter__(self):
        if not self._resumed:
            self._ptr = 0
        self._resumed = False
        try:
            from torch.utils.data.dataloader import DataLoader as _DataLoader
            from torch.utils.data.dataset import IterableDataset as _IterableDataset
            dataloader = _DataLoader(self._batch_iterator, batch_size=None, num_workers=self._num_workers,
                                     collate_fn=self._collate_fn, pin_memory=self._pin_memory, timeout=self._timeout,
                                     worker_init_fn=self._init_worker,
                                     multiprocessing_context=self._multiprocessing_context)
            for batch in itertools.islice(dataloader, self._ptr, None):
                self._ptr += 1
                yield batch

        except ImportError:
            if not self._warning:
                self._warning = True
                warnings.warn(
                    'Multi-worker dataloader requires PyTorch installation, since PyTorch is not available, '
                    'main thread will be used for loading data.')

            for batch in itertools.islice(self._batch_iterator, self._ptr, None):
                if self._collate_fn is not None:
                    batch = self._collate_fn(batch)
                self._ptr += 1
                yield batch
        self._ptr = 0


class _Queue(object):

    def __init__(self) -> None:
        self._queue = collections.deque()

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return self.qsize() == 0

    def push(self, x):
        self._queue.append(x)

    def pop(self):
        return self._queue.popleft()

    def pop_k(self, k=None):
        k = k or self.qsize()
        return [self.pop() for _ in range(k)]


def get_bucket_boundaries(inflation_rate: float = 1.1, inflation_offset: int = 0, min_length: int = 8,
                          max_length: int = 256, min_offset: int = 3):
    """
    Generate bucket boundaries to be used in BucketIterator. The value of inflation_rate and inflation_offset should
     be carefully picked to match the distribution of dataset. Greater value of inflation_rate or inflation_offset
     leads to sparse boundaries, which increases the chance of padding and thus reduces training efficiency;
     however, it also increases the diversity of samples in a batch and generate more diverse consecutive batches
     in terms of sample lengths. Sparse boundaries can avoid similar consecutive batches to some extent, which is
     bad for training.
    Args:
        inflation_rate:
        inflation_offset:
        min_length:
        max_length:
        min_offset:
    Returns:

    """
    if not (inflation_rate >= 1.0):
        raise ValueError(f'inflation_rate ({inflation_rate})  must be greater than 1.0.')
    if not (inflation_offset >= 0):
        raise ValueError(f'inflation_offset ({inflation_offset}) must be greater than 1.0.')

    x = min_length
    boundaries = []
    while x <= max_length:
        boundaries.append(x)
        x = max(x + min_offset, int(inflation_rate * x + inflation_offset))
    return boundaries


def _resize_batch_size(num_sample, required_batch_size_multiple):
    if num_sample <= required_batch_size_multiple:
        return num_sample
    return num_sample - (num_sample % required_batch_size_multiple)


class ConstantIterator(BatchIterator):
    """Constant-sized batch iterator.

    This iterator iterates over batches composed of a constant number of consecutive samples.

    """

    def __init__(self, data_source: Iterable, batch_size: int, drop_tail: bool = False) -> None:
        """Initialises the iterator.

        Args:
            data_source: a dataset as data source.
            batch_size: number of samples in a batch.
            drop_tail: whether to drop the trailing samples if they don't form a full batch.
        """
        super().__init__(data_source)
        self._batch_size = batch_size
        self._drop_tail = drop_tail
        self._num_batches = None

    def generator(self):
        it = iter(self._data_source)

        while True:
            samples = list(itertools.islice(it, self._batch_size))
            if len(samples) == self._batch_size:
                yield samples
            else:
                if not self._drop_tail and samples:
                    yield samples
                break

    def __len__(self):
        if self._num_batches is None:
            if self._drop_tail:
                self._num_batches = len(self._data_source) // self._batch_size
            else:
                self._num_batches = math.ceil(len(self._data_source) // self._batch_size)
        return self._num_batches


class BucketIterator(BatchIterator):
    """Varying-sized batch iterator.

    Varying-sized batch implies that the number of samples across different batches can differ. Since for each sample we can
    determine a particular size, for example, different text sequences have different number of tokens. In this regard,
    we define the batch size as the sum of sample sizes rather than the amount of samples.

    A range of buckets is created to hold similar sized samples into the same bucket for computational efficiency.
    Whenever a bucket reaches its capacity, it's yielded as a batch.
    """

    def __init__(
            self,
            data_source: Iterable,
            batch_size: int,
            get_length_fn: Callable[[Any], int],
            bucket_boundaries: List[int],
            min_length: int = 0,
            max_length: int = None,
            drop_tail: bool = False,
            required_batch_size_multiple: int = 1
    ) -> None:
        """Initialises the iterator.

        Args:
            data_source: a dataset object as data source.
            batch_size: batch size.
            get_length_fn: a function to evaluate the size of a given sample.
            bucket_boundaries: a list of ints that defines the buckets. Should be sorted in strictly ascending order.
            min_length: an int value indicates the minimum length of a sample, used as a filter.
            max_length: an int value indicates the maximum length of a sample, used as a filter.
            drop_tail: whether to drop the tail samples if they don't form a full batch.
            required_batch_size_multiple: adapts the actual batch size to be the multiple of this value.
            setting value to 8 to facilitate Tensor Core acceleration.

        """
        super().__init__(data_source)
        if not bucket_boundaries:
            raise ValueError(f'bucket_boundaries ({bucket_boundaries}) must be non-empty.')
        if not (bucket_boundaries == sorted(bucket_boundaries)):
            raise ValueError(f'bucket_boundaries ({bucket_boundaries}) must be in ascending order.')
        if not (len(set(bucket_boundaries)) == len(bucket_boundaries)):
            raise ValueError(f'bucket_boundaries ({bucket_boundaries}) must have unique elements.')
        if not callable(get_length_fn):
            raise ValueError(f'get_length_fn ({type(get_length_fn)}) must be callable.')
        if not (min_length <= bucket_boundaries[0]):
            raise ValueError(f'min_length ({min_length}) must be less than or equal to '
                             f'bucket_boundaries[0] ({bucket_boundaries[0]}).')
        if not (max_length is None or max_length >= bucket_boundaries[-1]):
            raise ValueError(f'max_length ({max_length}) must be less than or equal to '
                             f'bucket_boundaries[-1] ({bucket_boundaries[-1]}).')
        if not (required_batch_size_multiple > 0):
            raise ValueError(f'required_batch_size_multiple ({required_batch_size_multiple}) '
                             f'must be a positive integer.')

        max_length = max_length or bucket_boundaries[-1]

        self._get_length_fn = get_length_fn
        self._bucket_boundaries = bucket_boundaries
        self._batch_size = batch_size
        self._min_length = min_length
        self._max_length = max_length or bucket_boundaries[-1]
        self._drop_tail = drop_tail
        self._required_batch_size_multiple = required_batch_size_multiple

        if max_length > bucket_boundaries[-1]:
            bucket_boundaries.append(max_length)

        self._buckets = [_Queue() for _ in bucket_boundaries]
        self._capacities = [max(1, _resize_batch_size(batch_size // b, required_batch_size_multiple))
                            for b in bucket_boundaries]

    def generator(self):
        min_length = self._min_length
        max_length = self._max_length
        buckets = self._buckets
        capacities = self._capacities
        boundaries = self._bucket_boundaries
        num_bucket = len(buckets)
        len_fn = self._get_length_fn

        for x in self._data_source:
            size = len_fn(x)
            if size < min_length or size > max_length:
                continue
            i = bisect.bisect_left(boundaries, size)
            if i >= num_bucket:  # drop long sequence
                continue
            buckets[i].push(x)
            if buckets[i].qsize() == capacities[i]:
                yield buckets[i].pop_k(capacities[i])

        final_queue = _Queue()
        for q, capacity in zip(buckets, capacities):
            while not q.empty():
                final_queue.push(q.pop())
                if final_queue.qsize() >= capacity:
                    yield final_queue.pop_k(capacity)

        if not self._drop_tail and not final_queue.empty():
            yield final_queue.pop_k()

    def state(self):
        state = super().state()
        state['BucketIterator'] = {}
        state['BucketIterator']['_bucket_itr_queues'] = self._buckets
        state['BucketIterator']['_bucket_itr_capacities'] = self._capacities
        state['BucketIterator']['_bucket_itr_boundaries'] = self._bucket_boundaries
        return state

    def load(self, state):
        buckets = state['BucketIterator']['_bucket_itr_queues']
        capacities = state['BucketIterator']['_bucket_itr_capacities']
        bucket_boundaries = state['BucketIterator']['_bucket_itr_boundaries']
        if bucket_boundaries != self._bucket_boundaries:
            raise ValueError('Inconsistent bucket_boundaries.')
        if capacities != self._capacities:
            raise ValueError('Inconsistent bucket capacities.')
        self._buckets = buckets
        super().load(state)
