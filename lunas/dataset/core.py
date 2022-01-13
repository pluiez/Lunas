""" Abstract dataset classes and implementation of core datasets.
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
    'Sizable',
    'NestedN',
    'Nested',
    'NestedSizable',
    'Map',
    'Where',
    'Repeat',
    'Interleave',
    'Shuffle',
    'Sort',
    'Slice',
    'Shard',
    'Enumerate',
    'Group',
    'Concat'
]


class Dataset(abc.ABC):
    """An abstract class representing a dataset.

    A dataset is a wrapper for certain data source, providing interfaces to process and filter
    dataset elements during iteration, which are evaluated at runtime.

    The elements can be accessed through Python's iterator interface.
    """

    def __init__(self, name: str = None):
        """Initialises the dataset.
        Args:
            name: name of the dataset.
        """
        super().__init__()
        self._ptr = 0
        self._ref_count = 0
        self._resumed = False
        self._resumable = True

        self._name = name or self.__class__.__name__

    @abc.abstractmethod
    def generator(self):
        """A generator generates samples from the data source."""
        raise NotImplementedError

    def __iter__(self):
        """Returns an iterator of the dataset.

        Optionally, the iteration can be resumed by loading previously saved state, see `Dataset.state` and `Dataset.load`.
        """
        if not self._resumed:
            self._reset()

        it = self.generator()
        if self._resumable and self._ptr > 0:
            it = itertools.islice(it, self._ptr, None)
        self._resumed = False

        for x in it:
            self._ptr += 1
            if x is not None:
                yield x

        self._reset()

    @property
    def name(self):
        return self._name

    def state(self) -> dict:
        """Returns a checkpoint state

        The checkpoint can be used to resume iteration from previous stopping position.

        Returns:
            A dictionary containing necessary information to recover iteration state.
        """
        if not self._resumable:
            raise RuntimeError(f'dataset ({self.name}) is not resumable.')
        return {'ptr': self._ptr, 'name': self.name}

    def load(self, state: dict) -> None:
        """Recovers from a checkpoint state.

        Once recovered, we can create an iterator for the dataset to continue iteration from previous iteration.

        Args:
            state: a dictionary containing necessary information to recover iteration state.
        """
        if state is None:
            return

        if state['name'] != self.name:
            raise ValueError(f'name in state ({state["name"]}) should be the same as self.name ({self.name})')

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

    def map(self, fn: Callable[[Any], Any], unpack_args: bool = False, unpack_kwargs: bool = False,
            name: str = None) -> Map:
        """See `Map` class.

        Returns:
            A `Map` dataset.
        """
        return Map(self, fn, unpack_args, unpack_kwargs, name)

    def where(self, predicate: Callable[[Any], bool], unpack_args: bool = False, unpack_kwargs: bool = False,
              name: str = None) -> Where:
        """See `Where` class.

        Returns:
            A `Where` dataset.
        """
        return Where(self, predicate, unpack_args, unpack_kwargs, name)

    def repeat(self, n: int = None, name: str = None) -> Repeat:
        """See `Repeat` class.

        Returns:
            A `Repeat` dataset.
        """
        return Repeat(self, n, name)

    def shard(self, num_shards: int, shard_index: int, name: str = None) -> Shard:
        """See `Shard` class.

        Returns:
            A `Shard` dataset.
        """
        return Shard(self, num_shards, shard_index, name)

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

    def slice(self, start: int = None, stop: int = None, step: int = None, name: str = None) -> Slice:
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

    def group(self, group_size: int, name: str = None) -> Group:
        """See `Group` class.

        Returns:
            A `Group` dataset.
        """
        return Group(self, group_size, name)

    def flatten(self, name: str = None) -> Flatten:
        """See `Flatten` class.

        Returns:
            A `Flatten` dataset.
        """
        return Flatten(self, name)

    def concat(self, other: 'Dataset', name: str = None) -> Concat:
        """See `Concat` class.

        Returns:
            A `Concat` dataset.
        """
        return Concat([self, other], name)


class Sizable(abc.ABC):
    """An interface to evaluate length of a dataset."""

    @property
    def length(self):
        raise NotImplementedError

    def __len__(self):
        """Returns size of the dataset."""
        return self.length


class NestedN(Dataset, abc.ABC):
    """A wrapper around multiple datasets."""

    def __init__(self, datasets: Iterable[Dataset], name: str = None):
        super().__init__(name)
        if not isinstance(datasets, (tuple, list)):
            raise TypeError(f'datasets ({type(datasets)}) must be a tuple or a list.')
        datasets = tuple(datasets)
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError(f'datasets ({[type(dataset) for dataset in datasets]}) must inherit Dataset.')
        for x in datasets:
            x._chk_and_inc_ref()

        self._datasets: tuple = datasets
        self._resumable = all(dataset._resumable for dataset in datasets)

    @abc.abstractmethod
    def generator(self):
        """See base class."""
        raise NotImplementedError

    def _reset(self) -> None:
        super()._reset()
        for x in self._datasets:
            x._reset()


class Nested(NestedN, abc.ABC):
    """A wrapper around a single dataset."""

    def __init__(self, dataset: Dataset, name: str = None):
        super().__init__([dataset], name)
        self._dataset = dataset


class NestedSizable(Nested, Sizable, abc.ABC):
    """A sizable nested dataset."""

    def __init__(self, dataset: Dataset, name: str = None):
        super().__init__(dataset, name)
        self._length = None

    @property
    def length(self):
        if self._length is None:
            self._length = len(self._dataset)
        return self._length


class Map(NestedSizable):
    """Transform a sample.

    This dataset applies transformation to every sample in the dataset. Note that the transformation won't be applied until
    we try to access an element through an iterator.

    Example usage:

        ds = Range(100)
        ds = ds.map(lambda x: x * 2)
                .map(lambda x: x + 1)
        transformed = list(ds)
    """

    def __init__(self, dataset: Dataset, fn: Callable[[Any], Any], unpack_args: bool = False,
                 unpack_kwargs: bool = False, name: str = None) -> None:
        """Initialises the dataset.

        Args:
            dataset: the dataset to apply transformation.
            fn: a function that accepts a sample as input and returns a transformed output.
            unpack_args:
            unpack_kwargs:
            name:
        """
        super().__init__(dataset, name)
        if not callable(fn):
            raise ValueError(f'fn ({type(fn)}) must be callable.')
        if unpack_args and unpack_kwargs:
            raise ValueError(f'unpacking as args and kwargs are conflicting.')

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
    """Filter a sample by given predicate.

    This dataset applies filter to every sample in the dataset. Note that the filter won't be applied until we try to access
    an element through the iterator. A sample is only preserved if the predicate is evaluated True.

    Example usage:

        ds = Range(100)
        ds = ds.map(lambda x: x * 2)
                .where(lambda x: x % 10 == 0)
        transformed = list(ds)
    """

    def __init__(self, dataset: Dataset, predicate: Callable[[Any], bool], unpack_args: bool = False,
                 unpack_kwargs: bool = False, name: str = None):
        """
        Args:
            dataset: the dataset to apply filter.
            predicate: a function that accepts a sample as input and returns a `bool` value to indicate whether
            this sample should be preserved.
            unpack_args:
            unpack_kwargs:
            name:
        """
        super().__init__(dataset, name)
        if not callable(predicate):
            raise ValueError(f'predicate ({type(predicate)}) must be callable.')
        if unpack_args and unpack_kwargs:
            raise ValueError(f'unpacking as args and kwargs are conflicting.')

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


class Repeat(NestedSizable):
    """Repeat a dataset."""

    def __init__(self, dataset: Dataset, n: int = None, name: str = None):
        """Initialises the dataset.
        Args:
            dataset: a dataset to repeat.
            n: a value indicates how many times the dataset will repeat. Repeats endlessly if n is None.
            name:
        """
        super().__init__(dataset, name)
        if not (n is None or n > 0):
            raise ValueError(f'n ({n}) must be a positive integer or None.')

        self._n = n

    @property
    def length(self):
        if self._length is None:
            self._length = len(self._dataset) * self._n if self._n is not None else sys.maxsize
        return self._length

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
            dataset: a dataset object to generate subsets.
            map_fn: mapping element of `dataset` to a new dataset.
            cycle_length: number of consecutive datasets to access.
            block_length: number of consecutive elements to access.
            name: name of the dataset.
        """
        super().__init__(dataset.map(lambda x: iter(map_fn(x))), name)
        if not callable(map_fn):
            raise ValueError(f'map_fn ({type(map_fn)}) must be callable.')
        if not (cycle_length > 0):
            raise ValueError(f'cycle_length ({cycle_length}) must be a positive integer.')
        if not (block_length > 0):
            raise ValueError(f'block_length ({block_length}) must be a positive integer.')

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


class Shuffle(NestedSizable):
    """Shuffle a dataset.

    This dataset provides a queue-based shuffling to shuffle a dataset without loading all data into memory at once.
    The algorithm is an approximation of global shuffling.
    """

    def __init__(self, dataset: Dataset, buffer_size: int, name: str = None):
        """Initialises the dataset.

        Args:
            dataset: a dataset to shuffle.
            buffer_size: shuffles within the given buffer size.
            name:
        """
        super().__init__(dataset, name)
        if not (buffer_size > 1):
            raise ValueError(f'buffer_size ({buffer_size}) must be a positive integer greater than 1.')

        self._buffer_size = buffer_size

    def generator(self):
        it = iter(self._dataset)
        buffer = list(itertools.islice(it, self._buffer_size))
        buf_size = len(buffer)
        indices = numpy.nditer(numpy.random.randint(0, buf_size, buf_size))

        for x in it:
            try:
                i = next(indices)
            except StopIteration:
                indices = numpy.nditer(numpy.random.randint(0, buf_size, buf_size))
                i = next(indices)
            yield buffer[i]
            buffer[i] = x

        numpy.random.shuffle(buffer)

        for x in buffer:
            yield x


class Sort(NestedSizable):
    """Sort a dataset.

    This dataset provides a partial sorting to sort elements from a dataset in a custom-sized buffer, so that we don't
    load all data into memory at once. It is typically used to gather sequences of similar length together to reduce
    redundant computational overheads, while preserving randomness introduced by shuffling. To that end, we must use
    a larger buffer_size in Shuffle than Sort.

    For example:
        ds = TextLine(filename)
        ds = ds.shuffle(buffer_size=100000).sort(buffer_size=1000, key=lambda x: len(x.split()))
    """

    def __init__(self, dataset: Dataset, buffer_size: int, key: Callable[[Any], Any] = None, name: str = None):
        """Initialises the dataset.

        Args:
            dataset: a dataset to sort.
            buffer_size: sorts within the given buffer size.
            key: a function to get key for comparison.
            name:
        """
        super().__init__(dataset, name)
        if not (buffer_size > 1):
            raise ValueError(f'buffer_size ({buffer_size}) must be a positive integer greater than 1.')
        if not (key is None or callable(key)):
            raise ValueError(f'key ({type(key)}) must be callable.')

        self._buffer_size = buffer_size
        self._key = key

    def generator(self):
        it = iter(self._dataset)
        buf_size = self._buffer_size

        while True:
            buffer = sorted(itertools.islice(it, buf_size), key=self._key)

            for x in buffer:
                yield x

            if not buffer:
                break


class Slice(NestedSizable):
    """Slice a dataset."""

    def __init__(self, dataset: Dataset, start: int = 0, stop: int = None, step: int = 1, name: str = None):
        """Initialises the dataset

        Args:
            dataset: a dataset to slice.
            start: starting position.
            stop: stopping position.
            name:
        """
        super().__init__(dataset, name)
        if start is not None and not (start >= 0):
            raise ValueError(f'start ({start}) for Slice must be a non-negative integer or None.')
        if stop is not None and not (stop >= 0):
            raise ValueError(f'stop ({stop}) for Slice must be a non-negative integer or None.')
        if step is not None and not (step > 0):
            raise ValueError(f'step ({step}) for Slice must be a positive integer or None.')

        start = start or 0
        step = step or 1

        self._start = start
        self._stop = stop
        self._step = step

    @property
    def length(self):
        if self._length is None:
            self._length = max(0, math.ceil((len(self._dataset) - self._start) / self._step))
        return self._length

    def generator(self):
        for x in itertools.islice(self._dataset, self._start, self._stop, self._step):
            yield x


class Shard(Slice):
    """Shard the dataset.

    This dataset partitions elements alternately into several exclusive partitions and only one specific partition
    is loaded. This is typically used in distributed settings, where we want to ensure our training
    workers process different parts of the training data.

    For example, in worker-0:
        ds = TextLine(filename)
        ds = ds.shard(2, 0)

    And worker-1:
        ds = TextLine(filename)
        ds = ds.shard(2, 1)
    """

    def __init__(self, dataset: Dataset, num_shards: int, shard_index: int, name: str = None):
        """Initialises the dataset.
        Args:
            dataset: a dataset object to shard.
            num_shards: number of total shards.
            shard_index: index of current shard.
            name:
        """
        if not (shard_index >= 0):
            raise ValueError(f'shard_index ({shard_index}) must be an non-negative integer.')
        if not (shard_index < num_shards):
            raise ValueError(f'shard_index ({shard_index}) must be smaller than num_shards ({num_shards}).')

        super().__init__(dataset, shard_index, None, num_shards, name)


class Enumerate(NestedSizable):
    """Enumerate a dataset.

    This dataset simulates the builtin `enumerate` and attach an index to each element in the given dataset.
    """

    def __init__(self, dataset: Dataset, start: int = 0, name: str = None):
        super().__init__(dataset, name)
        self._start = start

    def generator(self):
        for x in enumerate(self._dataset, self._start):
            yield x


class Group(NestedSizable):
    """Group consecutive elements into an element."""

    def __init__(self, dataset: Dataset, group_size: int, name: str = None):
        if not (group_size >= 1):
            raise ValueError(f'group_size ({group_size}) must be a positive integer greater than 1.')

        super().__init__(dataset, name)

        self._group_size = group_size

    @property
    def length(self):
        if self._length is None:
            self._length = math.ceil(len(self._dataset) / self._group_size)
        return self._length

    def generator(self):
        it = iter(self._dataset)
        stopped = False
        while not stopped:
            group = []
            for _ in range(self._group_size):
                try:
                    group.append(next(it))
                except StopIteration:
                    stopped = True
                    break
            if group:
                yield group


class Flatten(Nested):
    """Flatten the element of a dataset into multiple elements.

    The size of this dataset can't be determined in advance because we have to flatten every element from the wrapped
    dataset to know the exact size after flattening.
    """

    def __init__(self, dataset: Dataset, name: str = None):
        super().__init__(dataset, name)

    def generator(self):
        for chunk in self._dataset:
            for x in chunk:
                yield x


class Concat(NestedN, Sizable):
    """Concat multiple datasets.

    This dataset is a concatenation of multiple datasets.
    """

    def __init__(self, datasets: Iterable[Dataset], name: str = None):
        super().__init__(list(datasets), name)
        self._length = None

    @property
    def length(self):
        if self._length is None:
            self._length = sum(map(len, self._datasets))
        return self._length

    def generator(self):
        for x in itertools.chain(*self._datasets):
            yield x
