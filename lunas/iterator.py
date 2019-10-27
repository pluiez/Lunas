import abc
import bisect
import itertools
from collections import deque
from typing import Callable, Any, List, Iterator, Iterable

from lunas.dataset.core import Dataset

__all__ = ['SimpleIterator', 'BucketIterator']


class BatchIterator(abc.ABC):

    def __init__(self, dataset: Dataset, batch_size: int, drop_tail=False) -> None:
        assert isinstance(dataset, Dataset)
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_tail = drop_tail

    @abc.abstractmethod
    def generator(self) -> Iterator:
        raise NotImplementedError

    def state_dict(self):
        return self._dataset.state_dict()

    def load_state_dict(self, state):
        self._dataset.load_state_dict(state)

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


class Queue(object):

    def __init__(self) -> None:
        self._queue = deque()

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return len(self._queue) == 0

    def put(self, x):
        self._queue.append(x)

    def get(self):
        return self._queue.popleft()

    def gets(self, n=None):
        n = n if n is not None else self.qsize()
        return [self.get() for _ in range(n)]


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def __init__(self, iterator: BatchIterator, num_workers: int = 0, collate_fn: Callable[[Iterable[Any]], Any] = None,
                 pin_memory: bool = False, timeout: int = 0, worker_init_fn: Callable[[None], None] = None,
                 multiprocessing_context=None): ...


try:
    from torch.utils.data.dataset import IterableDataset as _THIterableDataset
    from torch.utils.data.dataloader import DataLoader as _THDataLoader


    class _WrappedTHIterableDataset(BatchIterator, _THIterableDataset):
        ...


    class _WrappedTHDataLoader(DataLoader, _THDataLoader):
        # noinspection PyArgumentList
        def __init__(self, iterator: BatchIterator, num_workers: int = 0,
                     collate_fn: Callable[[Iterable[Any]], Any] = None,
                     pin_memory: bool = False, timeout: int = 0, worker_init_fn: Callable[[None], None] = None,
                     multiprocessing_context=None):
            _THDataLoader.__init__(self, iterator, batch_size=None, shuffle=False, sampler=None,
                                   batch_sampler=None, num_workers=num_workers, collate_fn=collate_fn,
                                   pin_memory=pin_memory, drop_last=False, timeout=timeout,
                                   worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context)
            self._lunas_iterator = iterator  # add additional reference simply for state persistence

        def state_dict(self):
            return self._lunas_iterator.state_dict()

        def load_state_dict(self, state):
            self._lunas_iterator.load_state_dict(state)

        def __iter__(self):
            return super().__iter__()


    BatchIterator = _WrappedTHIterableDataset
    DataLoader = _WrappedTHDataLoader
except ImportError:
    ...


class SimpleIterator(BatchIterator):
    def __init__(self, dataset: Dataset, batch_size: int, drop_tail: bool = False) -> None:
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


class BucketIterator(BatchIterator):
    def __init__(self, dataset: Dataset, batch_size: int,
                 sample_size_fn: Callable[[Any], int],
                 bucket_boundaries: List[int],
                 max_sample_size: int,
                 drop_tail: bool = False) -> None:
        """
        Arrange samples into different buckets determined by their size. This is useful for maximize the utilization
        of computation during a training step.
        :param dataset
        :param batch_size:
        :param sample_size_fn: a callable function, which evaluates the size of a given sample
        :param bucket_boundaries: a list of ints that defines the buckets' and accordingly their capacities. Should be
            sorted in ascending order
        :param max_sample_size: an optional integer. Samples with size exceeding this value will be discarded
        :param drop_tail:
        """
        super().__init__(dataset, batch_size, drop_tail)
        assert bucket_boundaries, 'bucket_boundaries should have at least one element'
        assert bucket_boundaries == sorted(bucket_boundaries), 'bucket_boundaries should be in ascending order'
        assert len(set(bucket_boundaries)) == len(bucket_boundaries), \
            'bucket_boundaries shall not contain duplicate elements'
        assert max_sample_size >= bucket_boundaries[
            -1], 'max_sample_size shall not be less than the last boundary of bucket_boundaries'
        assert callable(sample_size_fn), 'sample_size_fn should be a callable function'

        if max_sample_size > bucket_boundaries[-1]:
            bucket_boundaries.append(max_sample_size)

        self._sample_size_fn = sample_size_fn
        self._bucket_boundaries = bucket_boundaries
        self._max_sample_size = max_sample_size

    def generator(self):
        boundaries = self._bucket_boundaries

        buckets = [Queue() for _ in boundaries]
        batch_sizes = [max(1, self._batch_size // b) for b in boundaries]

        num_bucket = len(buckets)

        sample_size_fn = self._sample_size_fn

        for x in self._dataset:
            size = sample_size_fn(x)
            i = bisect.bisect_left(boundaries, size)

            if i >= num_bucket:  # drop sample with size exceeding the max boundary
                continue

            buckets[i].put(x)
            if buckets[i].qsize() == batch_sizes[i]:
                yield i, buckets[i].gets(batch_sizes[i])

        if not self._drop_tail:
            final_queue = Queue()
            for q, size in zip(buckets, batch_sizes):
                while not q.empty():
                    final_queue.put(q.get())
                    if final_queue.qsize() >= size:
                        yield final_queue.gets(2 * size - final_queue.qsize())

            if not final_queue.empty():
                yield final_queue.gets()
