"""
This module defines iterators to arrange a list of samples as a batch and process the samples to be fed into a model.
"""

import abc
import bisect
import collections
import itertools
import logging
from typing import Callable, Any, List, Iterator, Union

import lunas.dataset.core as core

__all__ = ['ConstantIterator', 'BucketIterator', 'DataLoader', 'get_bucket_boundaries']

try:
    from torch.utils.data.dataset import IterableDataset as _IterableDataset
except ImportError:
    class _IterableDataset(object):
        ...


class _BatchIterator(abc.ABC, _IterableDataset):

    def __init__(self, dataset: core.Dataset, batch_size: int, drop_tail=False) -> None:
        """Initialises the iterator.

        Args:
            dataset: A dataset object as data source.
            batch_size: Batch size.
            drop_tail: Whether to drop the tail samples if they don't form a full batch.
        """
        if not isinstance(dataset, core.Dataset):
            raise TypeError(f'invalid dataset with type: "{type(dataset)}"')
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_tail = drop_tail

    @abc.abstractmethod
    def generator(self) -> Iterator:
        raise NotImplementedError

    def state(self):
        return self._dataset.state()

    def load(self, state):
        self._dataset.load(state)

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


def get_bucket_boundaries(max_length=512, min_length=8, length_bucket_step=1.1):
    """Returns a list of bucket boundaries.

    Bucket boundaries are necessary in BucketIterator to arrange similar sized samples into corresponding bucket.

    Args:
        max_length: The upper bound. Sample with size greater than this value will be dropped.
        min_length: The lower bound. Sample with size less than or equal to this value will be classified into first
            bucket (i.e., 0-th bucket).
        length_bucket_step: Determines how much next boundary is scaled proportional to the previous one.
    """
    if not (length_bucket_step > 1.0):
        raise ValueError(f'length_bucket_step must be greater than 1.0: {length_bucket_stepp}')

    x = min_length
    boundaries = []
    while x < max_length:
        boundaries.append(x)
        x = max(x + 1, int(x * length_bucket_step))
    return boundaries


class DataLoader(object):
    """Loads and batches data.

    DataLoader wraps around a BatchIterator instance, batches multiple samples and optionally moves the data to pinned
    memory.

    This DataLoader relies on PyTorch's carefully designed DataLoader to implement secure multi-processing operations.
    Since in a naive implementation, the main Python process can lose connection with one of the multiprocessing
    process if it crashed, for example, by segment fault. However, PyTorch's DataLoader deals with this problem by
    tracking each process's states manually. Here we can reuse it for simplicity.
    """

    def __init__(
            self,
            batch_iterator: _BatchIterator,
            num_workers: int = 0,
            collate_fn: Callable[[List], Any] = None,
            pin_memory: bool = True,
            timeout: int = 0,
            worker_init_fn=None,
            multiprocessing_context=None
    ):
        if not isinstance(batch_iterator, _BatchIterator):
            raise TypeError(f'invalid instance with type: "{type(batch_iterator)}"')
        self._batch_iterator = batch_iterator
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._timeout = timeout
        self._worker_init_fn = worker_init_fn
        self._multiprocessing_context = multiprocessing_context
        self._warning = False

    def state(self):
        return self._batch_iterator.state()

    def load(self, state):
        self._batch_iterator.load(state)

    def __iter__(self):
        try:
            from torch.utils.data.dataloader import DataLoader as _DataLoader
            from torch.utils.data.dataset import IterableDataset as _IterableDataset

            dataloader = _DataLoader(self._batch_iterator, batch_size=None, num_workers=self._num_workers,
                                     collate_fn=self._collate_fn, pin_memory=self._pin_memory, timeout=self._timeout,
                                     worker_init_fn=self._worker_init_fn,
                                     multiprocessing_context=self._multiprocessing_context)
            for batch in dataloader:
                yield batch
        except ImportError:
            if not self._warning:
                self._warning = True
                logging.warning(
                    'Multi-worker dataloader requires PyTorch installation, since PyTorch is not available, '
                    'main thread will be used for loading data.')

            for batch in self._batch_iterator:
                if self._collate_fn is not None:
                    batch = self._collate_fn(batch)
                yield batch


class ConstantIterator(_BatchIterator):
    """Constant-sized batch iterator.

    Iterates by a constant number of samples for each batch.

    """

    def __init__(self, dataset: core.Dataset, batch_size: int, drop_tail: bool = False) -> None:
        """Initialises the iterator.

        Args:
            dataset: A dataset object as data source.
            batch_size: Number of samples in a batch.
            drop_tail: Whether to drop the tail samples if they don't form a full batch.
        """
        super().__init__(dataset, batch_size, drop_tail)

    def generator(self):
        it = iter(self._dataset)

        while True:
            samples = list(itertools.islice(it, self._batch_size))
            if len(samples) == self._batch_size:
                yield samples
            else:
                if not self._drop_tail and samples:
                    yield samples
                break


class BucketIterator(_BatchIterator):
    """Varying-sized batch iterator.

    Varying-sized batch implies the number of samples across different batches can differ. Since for each sample we can
    determine a particular size, for example, different text sequences have different number of tokens. In this regard,
    we define the batch size as the sum of sample sizes rather than the amount of samples.

    A range of buckets is created to hold similar sized samples into the same bucket for computational efficiency.
    Whenever a bucket reaches its capacity, it's yielded as a batch.
    """

    def __init__(
            self,
            dataset: core.Dataset,
            batch_size: int,
            get_length_fn: Callable[[Any], int],
            bucket_boundaries: List[int],
            min_length: Union[None, int] = None,
            max_length: Union[None, int] = None,
            drop_tail: bool = False
    ) -> None:
        """Initialises the iterator.

        Args:
            dataset: A dataset object as data source.
            batch_size: Batch size.
            get_length_fn: A function to evaluate the size of a given sample.
            bucket_boundaries: A list of ints that defines the buckets. Should be sorted in strictly ascending order.
            min_length: Samples with size < min_length will be discarded.
            max_length: Samples with size > max_length will be discarded.
            drop_tail: Whether to drop the tail samples if they don't form a full batch.
        """
        super().__init__(dataset, batch_size, drop_tail)
        if not bucket_boundaries:
            raise ValueError(f'bucket_boundaries must have at least one element: {bucket_boundaries}')
        if not (bucket_boundaries == sorted(bucket_boundaries)):
            raise ValueError(f'bucket_boundaries must be in ascending order: {bucket_boundaries}')
        if not (len(set(bucket_boundaries)) == len(bucket_boundaries)):
            raise ValueError(f'bucket_boundaries must not contain duplicate elements: {bucket_boundaries}')
        if not (max_length is None or max_length > bucket_boundaries[-1]):
            raise ValueError(f'max_length must be None or a value greater than the last bucket_boundary: '
                             f'({max_length}, {bucket_boundaries[-1]})')
        if not (min_length is None or min_length < bucket_boundaries[0]):
            raise ValueError(f'min_length must be None or a value smaller than the first bucket_boundary: '
                             f'({min_length}, {bucket_boundaries[0]})')
        if not callable(get_length_fn):
            raise ValueError(f'instance of type "{type(get_length_fn)}" is not callable.')

        if max_length is not None:
            bucket_boundaries.append(max_length)
        if min_length is not None:
            bucket_boundaries.insert(0, min_length)

        self._get_length_fn = get_length_fn
        self._bucket_boundaries = bucket_boundaries
        self._min_length = min_length
        self._max_length = max_length

    def generator(self):
        boundaries = self._bucket_boundaries

        buckets = [_Queue() for _ in boundaries]
        capacities = [max(1, self._batch_size // b) for b in boundaries]

        num_bucket = len(buckets)

        len_fn = self._get_length_fn

        for x in self._dataset:
            size = len_fn(x)
            # drop short sequence
            if self._min_length is not None and size < self._min_length:
                continue

            i = bisect.bisect_left(boundaries, size)

            if i >= num_bucket:  # drop long sequence
                continue

            buckets[i].push(x)
            if buckets[i].qsize() == capacities[i]:
                yield buckets[i].pop_k(capacities[i])

        if not self._drop_tail:
            final_queue = _Queue()
            for q, capacity in zip(buckets, capacities):
                while not q.empty():
                    final_queue.push(q.pop())
                    if final_queue.qsize() >= capacity:
                        yield final_queue.pop_k(capacity)

            if not final_queue.empty():
                yield final_queue.pop_k()
