""" Abstract dataset class and datasets for generic use.
"""

from __future__ import annotations

import abc
import itertools
import math
import sys
from typing import *

import numpy

__all__ = [
    'Dataset',
    'Map',
    'Where',
    'Repeat',
    'Interleave',
    'Shuffle',
    'Sort',
    'Slice',
    'Shard',
    'Enumerate',
    'Chunk',
    'Concat'
]


class Dataset(abc.ABC):
    """An abstract class representing a dataset.

    A dataset is a wrapper for certain data sources, providing interfaces to transformation and filtration on
    the samples, which are evaluated lazily.

    The elements can be accessed through Python's iterator interface.
    """

    def __init__(self, name: str = None):
        """Initialises the dataset.
        Args:
            name: Name of this dataset, useful for providing debugging information.
        """
        super().__init__()
        self._fns: List = []
        self._ptr = 0
        self._name = name or self.__class__.__name__
        self._ref_count = 0
        self._resumed = False
        self._resumable = True

    @abc.abstractmethod
    def __len__(self):
        """Returns size of the dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def generator(self):
        """Yields a single sample from data source."""
        raise NotImplementedError

    def __iter__(self):
        """Returns an iterator of the dataset.

        Continues iteration if the dataset instance just resumed from a saved state.
        Otherwise, re-iterates from the beginning of the dataset.
        """
        if len(self) == 0:
            return
        ptr = self._ptr % len(self)
        if not self._resumed:
            ptr = 0
        it = self.generator()
        if self._resumable:
            it = itertools.islice(it, ptr, None)
        self._resumed = False
        for x in it:
            self._ptr += 1
            if x is not None:
                yield x

    @property
    def name(self):
        return self._name

    def state(self) -> dict:
        """Returns a checkpoint state

        The checkpoint can be used to resumes iteration from previous stopping point.

        Returns:
            A dictionary containing necessary information to recover iteration state later.
        """
        if not self._resumable:
            raise RuntimeError(f'Instance of {type(self)} is not resumable.')
        return {'ptr': self._ptr, 'name': self._name}

    def load(self, state: dict) -> None:
        """Recovers from a checkpoint state.

        Once recovered, we can create an iterator for the dataset to continue iteration from previous iteration.

        Args:
            state: A dictionary containing necessary information to recover iteration state.
        """
        if state is None:
            return

        if state['name'] != self._name:
            raise ValueError(f'name in state ({state["name"]}) should be the same as self._name ({self._name})')

        self._ptr = state['ptr']
        self._resumed = True

    def _chk_and_inc_ref(self) -> None:
        """Ensures a dataset to be referenced by at most 1 dataset."""
        if self._ref_count > 0:
            raise RuntimeError(f'Dataset `{self}` is only allowed to be wrapped in another dataset once, '
                               'you should consider create a new instance instead.')
        else:
            self._ref_count += 1

    def _reset(self) -> None:
        """Resets iteration state."""
        self._ptr = 0
        self._resumed = False

    def map(self, fn: Callable[[Any], Any], unpack_args=False, unpack_kwargs=False, name: str = None) -> Map:
        if not callable(fn):
            raise ValueError(f'instance of type "{type(fn)}" is not callable.')
        if unpack_args and unpack_kwargs:
            raise ValueError(f'Cannot unpack as args and kwargs at the same time.')
        return Map(self, fn, unpack_args, unpack_kwargs, name)

    def where(self, predicate: Callable[[Any], bool], unpack_args=False, unpack_kwargs=False,
              name: str = None) -> Where:
        if not callable(predicate):
            raise ValueError(f'instance of type "{type(predicate)}" is not callable.')
        if unpack_args and unpack_kwargs:
            raise ValueError(f'Cannot unpack as args and kwargs at the same time.')
        return Where(self, predicate, unpack_args, unpack_kwargs, name)

    def repeat(self, n: int = None, name: str = None) -> Repeat:
        """See `Repeat` class.

        Returns:
            A `Repeat` dataset.
        """
        return Repeat(self, n, name)

    def shard(self, num_shards: int, index: int, name: str = None) -> Shard:
        """See `Shard` class.

        Returns:
            A `Shard` dataset.
        """
        return Shard(self, num_shards, index, name)

    def shuffle(self, buffer_size: int, name: str = None) -> Shuffle:
        """See `Shuffle` class.

        Returns:
            A `Shuffle` dataset.
        """
        return Shuffle(self, buffer_size, name)

    def sort(self, buffer_size: int, key: Callable[[Any], Any] = None, name: str = None) -> Sort:
        """See `Sort` class.

        Returns:
            A `Sort` dataset.
        """
        return Sort(self, buffer_size, key, name)

    def slice(self,
              start: Optional[int] = None,
              stop: Optional[int] = None,
              step: Optional[int] = None,
              name: str = None) -> Slice:
        """See `Slice` class.

        Returns:
            A `Slice` dataset.
        """
        return Slice(self, start, stop, step, name)

    def take(self, n: int, name: str = None) -> Slice:
        """See `Take` class.

        Returns:
            A `Take` dataset.
        """
        return Slice(self, stop=n, name=name)

    def skip(self, n: int, name: str = None) -> Slice:
        """See `Skip` class.

        Returns:
            A `Skip` dataset.
        """
        return Slice(self, start=n, name=name)

    def enumerate(self, start: int = 0, name: str = None) -> Enumerate:
        """See `Enumerate` class.

        Returns:
            A `Enumerate` dataset.
        """
        return Enumerate(self, start, name)

    def chunk(self, chunk_size: int, name: str = None) -> Chunk:
        """See `Chunk` class.

        Returns:
            A `Chunk` dataset.
        """
        return Chunk(self, chunk_size, name)

    def concat(self, other: 'Dataset', name: str = None) -> Concat:
        """See `Concat` class.

        Returns:
            A `Concat` dataset.
        """
        return Concat([self, other], name)


class NestedN(Dataset):
    """A wrapper for multiple datasets."""

    def __init__(self, datasets: Iterable[Dataset], name: str = None):
        super().__init__(name)
        if not isinstance(datasets, (tuple, list)):
            raise TypeError(f'datasets must be a tuple or a list: {type(datasets)}')
        datasets = tuple(datasets)
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError(f'datasets must subclass Dataset. '
                                f'Got: {[type(dataset) for dataset in datasets]}')
        for x in datasets:
            x._chk_and_inc_ref()

        self._datasets: tuple = datasets
        self._resumable = all(dataset._resumable for dataset in datasets)

    @abc.abstractmethod
    def __len__(self):
        """See base class."""
        raise NotImplementedError

    @abc.abstractmethod
    def generator(self):
        """See base class."""
        raise NotImplementedError

    def _reset(self) -> None:
        """See base class."""
        super()._reset()
        for x in self._datasets:
            x._reset()


class Nested(NestedN):
    """A wrapper for one dataset."""

    def __init__(self, dataset: Dataset, name: str = None):
        super().__init__([dataset], name)
        self._dataset = dataset

    @abc.abstractmethod
    def generator(self):
        """See base class."""
        raise NotImplementedError

    def __len__(self):
        """See base class."""
        return len(self._dataset)


class Map(Nested):
    """Transforms a sample.

    Applies transformation to every sample in the dataset. Note that the transformation won't be applied until
    we try to access an element through an iterator.

    Example usage:

        ds = Range(100)
        ds = ds.map(lambda x: x * 2)
                .map(lambda x: x + 1)
        transformed = list(ds)
    """

    def __init__(self, dataset: Dataset, fn: Callable, unpack_args=False, unpack_kwargs=False,
                 name: str = None) -> None:
        """Initialises the dataset.

        Args:
            dataset: The dataset object to apply transformation.
            fn: A function that accepts a sample as input and returns a transformed one.
            unpack_args:
            unpack_kwargs:
            name:
        """
        super().__init__(dataset, name)
        if unpack_args and unpack_kwargs:
            raise ValueError(f'Cannot unpack as args and kwargs at the same time.')
        self._fn = fn
        self._unpack_args = unpack_args
        self._unpack_kwargs = unpack_kwargs

    def generator(self):
        if self._unpack_args:
            for x in self._dataset:
                yield self._fn(*x)
        elif self._unpack_kwargs:
            for x in self._dataset:
                yield self._fn(**x)
        else:
            for x in self._dataset:
                yield self._fn(x)


class Where(Nested):
    """Filters a sample by given predicate.

    Applies filter to every sample in the dataset. Note that the filter won't be applied until we try to access
    an element through the iterator.

    Example usage:

        ds = Range(100)
        ds = ds.map(lambda x: x * 2)
                .where(lambda x: x % 10 == 0)
        transformed = list(ds)
    """

    def __init__(self, dataset: Dataset, predicate: Callable[[Any], bool], unpack_args=False,
                 unpack_kwargs=False, name: str = None):
        """

        Args:
            dataset: The dataset object to apply filter.
            predicate: A function that accepts a sample as input and returns a `bool` value to indicate whether
            this sample should be dropped.
            unpack_args:
            unpack_kwargs:
            name:
        """
        super().__init__(dataset, name)
        if unpack_args and unpack_kwargs:
            raise ValueError(f'Cannot unpack as args and kwargs at the same time.')
        self._predicate = predicate
        self._unpack_args = unpack_args
        self._unpack_kwargs = unpack_kwargs

    def generator(self):
        if self._unpack_args:
            for x in self._dataset:
                if self._predicate(*x):
                    yield x
        elif self._unpack_kwargs:
            for x in self._dataset:
                if self._predicate(**x):
                    yield x
        else:
            for x in self._dataset:
                if self._predicate(x):
                    yield x


class Repeat(Nested):
    """Repeats a dataset."""

    def __init__(self, dataset: Dataset, n: int = None, name: str = None):
        """Initialises the dataset.
        Args:
            dataset: A dataset object to repeat.
            n: A value indicates how many times the dataset will repeats. Repeats endlessly if provided `None`.
            name: Name of the dataset.
        """
        super().__init__(dataset, name)
        if not (n is None or n > 0):
            raise ValueError(f'n must be greater than 0: {n}')
        self._n = n

    def __len__(self):
        return len(self._dataset) * self._n if self._n is not None else sys.maxsize

    def generator(self):
        repeats = itertools.repeat(self._dataset, self._n) if self._n else itertools.repeat(self._dataset)
        for x in itertools.chain.from_iterable(repeats):
            yield x


class Interleave(Nested):
    """Interleave iteration between multiple datasets.

    map_fn maps each element in dataset to a new dataset, then we interleave iteration according to cycle_length and
    block_length. By cycle_length, we iterate over the first cycle_length datasets alternatively until any of which
    is exhausted and then switch to the next available dataset. At each cycle, we sequentially consumed block_length
    examples from current dataset.

    One use case of this class, is partitioning a large dataset into multiple directories and shuffling each subset
    within the directory independently, then interleave access to different subsets to approximate global shuffling
    of the large dataset without re-shuffling the whole dataset, which can be rather time-consuming.

    For example:
        directories = Glob('path/to/dataset')
        ds = InterleaveDataset(directories, lambda filename: TextLine(filename), 3, 1000)
    """

    def __init__(self, dataset: Dataset, map_fn: Callable[[Any], Dataset], cycle_length: int, block_length: int,
                 name: str = None):
        """Initialises the dataset.

        Args:
            dataset: A dataset object to generate subsets.
            map_fn: Mapping element of `dataset` to a new dataset.
            cycle_length: Number of consecutive datasets to access.
            block_length: Number of consecutive elements to access.
            name: Name of the dataset.
        """
        super().__init__(dataset.map(lambda x: iter(map_fn(x))), name)
        if not callable(map_fn):
            raise ValueError(f'instance of type "{type(map_fn)}" is not callable.')
        if not (cycle_length > 0):
            raise ValueError(f'cycle_length must be greater than 0: {cycle_length}')
        if not (block_length > 0):
            raise ValueError(f'block_length must be greater than 0: {block_length}')
        self._cycle_length = cycle_length
        self._block_length = block_length

    def generator(self):
        it = iter(self._dataset)
        xs = [x for x in itertools.islice(it, self._cycle_length)]
        i = 0
        cycle_length = len(xs)
        while cycle_length > 0:
            i = i % cycle_length
            x = xs[i]
            items = list(itertools.islice(x, self._block_length))
            if not items:
                xs.pop(i)
                try:
                    xs.append(next(it))
                except StopIteration:
                    i -= 1
                    cycle_length -= 1
            else:
                for item in items:
                    yield item
                i += 1


class Shuffle(Nested):
    """Shuffles the dataset.

    A queue-based shuffler to shuffle incoming input stream without loading all data into memory at once. The algorithm
    is an approximation of global shuffling.
    """

    def __init__(self, dataset: Dataset, buffer_size: int, name: str = None):
        """Initialises the dataset.

        Args:
            dataset: A dataset object to shuffle.
            buffer_size: Shuffles within the buffer size.
            name: Name of the dataset.
        """
        super().__init__(dataset, name)
        if not (buffer_size > 1):
            raise ValueError(f'buffer_size must be greater than 1: {buffer_size}')
        self._buffer_size = buffer_size

    def generator(self):
        it = iter(self._dataset)
        size = min(self._buffer_size, len(self._dataset))
        buffer = list(itertools.islice(it, size))
        indices = numpy.nditer(numpy.random.randint(0, size, size))

        for x in it:
            try:
                i = next(indices)
            except StopIteration:
                indices = numpy.nditer(numpy.random.randint(0, size, size))
                i = next(indices)
            yield buffer[i]
            buffer[i] = x

        numpy.random.shuffle(buffer)
        for x in buffer:
            yield x


class Sort(Nested):
    """Sorts the dataset.

    A partial sorting that sorts elements from the dataset in a custom-sized buffer so that we don't load all data
    into memory at once. Typically used to gather sequences of similar length together to reduce redundant
    computational overheads, while preserving randomness introduced by shuffling. To that end, we must use a larger
    buffer_size in Shuffle than Sort.

    For example:
        ds = TextLine(filename)
        ds = ds.shuffle(buffer_size=100000).sort(buffer_size=1000, key=lambda x: len(x.split()))
    """

    def __init__(self, dataset: Dataset, buffer_size: int, key: Callable[[Any], Any] = None, name: str = None):
        """Initialises the dataset.

        Args:
            dataset: A dataset object to sort.
            buffer_size: Sorts within this number of samples.
            key: A function to get key for comparison.
            name: Name of the dataset.
        """
        super().__init__(dataset, name)
        if not (buffer_size > 1):
            raise ValueError(f'buffer_size must be greater than 1: {buffer_size}')
        if not (key is None or callable(key)):
            raise ValueError(f'instance of type "{key}" is not callable.')
        self._buffer_size = buffer_size
        self._key = key

    def __len__(self):
        return len(self._dataset)

    def generator(self):
        it = iter(self._dataset)
        size = min(self._buffer_size, len(self._dataset))

        while True:
            buffer = sorted(itertools.islice(it, size), key=self._key)
            for x in buffer:
                yield x

            if not buffer:
                break


class Slice(Nested):
    """Slices the dataset."""

    def __init__(self,
                 dataset: Dataset,
                 start: int = 0,
                 stop: Optional[int] = None,
                 step: int = 1,
                 name: str = None):
        """Initialises the dataset

        Args:
            dataset: A dataset object to slice.
            start: Starting position.
            stop: Stopping position.
            name: Name of the dataset.
        """
        super().__init__(dataset, name)
        if start is not None and not (start >= 0):
            raise ValueError(f'start for Slice must be a non-negative integer or None, got {start} instead.')
        if stop is not None and not (stop >= 0):
            raise ValueError(f'stop for Slice must be a non-negative integer or None, got {stop} instead.')
        if step is not None and not (step > 0):
            raise ValueError(f'step for Slice must be a positive integer or None, got {step} instead.')

        start = start or 0
        stop = len(dataset) if stop is None else min(len(dataset), stop)
        step = step or 1

        self._size = max(0, math.ceil((stop - start) / step))
        self._start = start
        self._stop = stop
        self._step = step

    def __len__(self):
        return self._size

    def generator(self):
        it = iter(self._dataset)
        it = itertools.islice(it, self._start, self._stop, self._step)
        for x in it:
            yield x

    def state(self) -> dict:
        state = super().state()
        return state

    def load(self, state: dict) -> None:
        super().load(state)

    def _reset(self) -> None:
        super()._reset()


class Shard(Slice):
    """Shards the dataset.

    Sharding means we split the dataset sequentially into several mutual exclusive partitions, and loads only one of
    the specific partition. This is typically used in distributed settings, where we want to ensure our training
    workers process different parts of the training data.

    For example in worker-0:
        ds = TextLine(filename)
        ds = ds.shard(2, 0)

    And worker-1:
        ds = TextLine(filename)
        ds = ds.shard(2, 1)
    """

    def __init__(self, dataset: Dataset, num_shards: int, index: int, name: str = None):
        """Initialises the dataset.
        Args:
            dataset: A dataset object to shard.
            num_shards: Number of total shards.
            index: Index of current shard.
            name: Name of the sharded dataset.
        """
        if not (index >= 0):
            raise ValueError(f'index ({index}) must be an non-negative integer.')
        if not (index < num_shards):
            raise ValueError(f'index ({index}) must be smaller than num_shards ({num_shards}).')
        if len(dataset) < num_shards:
            raise ValueError(f'num_shards ({num_shards}) must be less than or equal to dataset size ({len(dataset)}).')
        super().__init__(dataset, index, None, num_shards)


class Enumerate(Nested):
    """Enumerate a dataset.

    Simulates the builtin enumerate function and attach an index to each element in the given dataset.
    """

    def __init__(self, dataset: Dataset, start: int = 0, name: str = None):
        super().__init__(dataset, name)
        self._start = start

    def __len__(self):
        return len(self._dataset)

    def generator(self):
        for x in enumerate(self._dataset, self._start):
            yield x


class Chunk(Nested):

    def __init__(self, dataset: Dataset, chunk_size: int, name: str = None):
        if not (chunk_size >= 1):
            raise ValueError(f'chunk_size ({chunk_size}) must be greater than 1.')
        super().__init__(dataset, name)
        self._chunk_size = chunk_size

    def generator(self):
        it = iter(self._dataset)
        stopped = False
        while not stopped:
            chunk = []
            for _ in range(self._chunk_size):
                try:
                    chunk.append(next(it))
                except StopIteration:
                    stopped = True
                    break
            if chunk:
                yield chunk


class Concat(NestedN):
    """Concat dataset.

    Concatenates multiple datasets.
    """

    def __init__(self, datasets: Iterable[Dataset], name: str = None):
        super().__init__(list(datasets), name)

    def __len__(self):
        return sum(map(len, self._datasets))

    def generator(self):
        for x in itertools.chain(*self._datasets):
            yield x
