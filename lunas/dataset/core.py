from __future__ import annotations

import abc
import inspect
import itertools
import sys
from typing import List, Callable, Any

import numpy


class Dataset(abc.ABC):
    """An abstract class representing a dataset.

    A dataset includes pipeline to process each sample and iterator to scan through the dataset.
    This class defines any interface that's visible to users.
    """

    def __init__(self, name: str = None):
        super().__init__()
        self._fns: List = []
        self._ptr = 0
        self._name = name

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def generator(self):
        raise NotImplementedError

    @property
    def name(self):
        return self.name

    def __iter__(self):
        ptr = self._ptr % len(self)
        for self._ptr, x in enumerate(itertools.islice(self.generator(), ptr, len(self)), ptr + 1):
            x = self._apply(x)
            if x is not None:
                yield x

    def state_dict(self):
        return {'ptr': self._ptr}

    def load_state_dict(self, state):
        self._ptr = state['ptr']

    def select(self, fn: Callable[[Any], Any]) -> Dataset:
        """Transforms a sample lazily.

        fn is a callable object that takes a sample as input and outputs a transformed
        sample.

        Args:
            fn: A `Callable` object.

        Returns:
            `self`.

        """
        assert callable(fn)
        self._fns.append(fn)
        return self

    def where(self, predicate: Callable[[Any], bool]) -> Dataset:
        """Filters a sample by predicate.

        predicate is a callable object that takes a sample as input and outputs a
        `bool` value to indicate whether this sample should be dropped.

        Args:
            predicate: A `Callable` object.

        Returns:
            `self`.
        """
        assert callable(predicate)
        self._fns.append(predicate)
        return self

    def _apply(self, sample: Any) -> Any:
        """Applies transformations and filters to a sample.

        When sample is filtered out, returns None, else returns transformed sample.

        Args:
            sample: An input sample to be processed.

        Returns:
            A sample or `None` if it's filtered.
        """
        if sample is None:
            return None

        unpack_args = isinstance(sample, (tuple, list))

        for fn in self._fns:
            try:
                if unpack_args:
                    new_sample = fn(*sample)
                else:
                    new_sample = fn(sample)
            except TypeError as e:
                raise Exception(f'Incompatible mapping function and inputs in `{type(self)}`.'
                                f'Function signature: {inspect.signature(fn)}.'
                                f'Inputs: {sample}.') from e

            # Stops when predicate is evaluated to False.
            if new_sample is False:
                return None

            if new_sample is not True:
                sample = new_sample
        return sample

    def concat(self, other: Dataset) -> Concat:
        return Concat(self, other)

    def repeat(self, n=None, name: str = None) -> Repeat:
        return Repeat(self, n, name)

    def shard(self, num_shards: int, index: int, name: str = None) -> Shard:
        return Shard(self, num_shards, index, name)

    def shuffle(self, buffer_size: int, name: str = None) -> Shuffle:
        return Shuffle(self, buffer_size, name)

    def interleave(self, map_fn: Callable[[Any], Dataset], cycle_length: int, block_length: int,
                   name: str = None) -> InterleaveDataset:
        return InterleaveDataset(self, map_fn, cycle_length, block_length, name)

    def sort(self, buffer_size: int, key: Callable[[Any], Any] = None, name: str = None) -> Sort:
        return Sort(self, buffer_size, key, name)

    def take(self, n: int, name: str = None) -> Take:
        return Take(self, n, name)

    def skip(self, n: int, name: str = None) -> Skip:
        return Skip(self, n, name)

    def window(self, size: int, shift: int = None, stride: int = 1,
               drop_tail: bool = False, name: str = None) -> Window:
        return Window(self, size, shift, stride, drop_tail, name)


class Nested(Dataset):

    def __init__(self, dataset: Dataset, name: str = None):
        super().__init__(name)
        assert isinstance(dataset, Dataset)
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def generator(self):
        raise NotImplementedError


class Repeat(Nested):

    def __init__(self, dataset: Dataset, n: int = None, name: str = None):
        super().__init__(dataset, name)
        assert n is None or n > 0
        self._n = n

    def __len__(self):
        return len(self._dataset) * self._n if self._n is not None else sys.maxsize

    def load_state_dict(self, state):
        self._ptr = state['ptr'] % len(self._dataset)

    def generator(self):
        for _ in itertools.repeat(None, self._n) if self._n else itertools.repeat(None):
            for x in self._dataset:
                yield x


class InterleaveDataset(Nested):

    def __init__(self, dataset: Dataset, map_fn: Callable[[Any], Dataset], cycle_length: int, block_length: int,
                 name: str = None):
        dataset.select(lambda x: iter(map_fn(x)))
        super().__init__(dataset, name)
        assert callable(map_fn)
        assert cycle_length > 0
        assert block_length > 0
        self._cycle_length = cycle_length
        self._block_length = block_length

    def generator(self):
        it = iter(self._dataset)
        xs = [iter(x) for x in itertools.islice(it, self._cycle_length)]
        i = 0
        cycle_length = len(xs)
        while cycle_length > 0:
            j = i % cycle_length
            x = xs[j]
            items = list(itertools.islice(x, self._block_length))
            if not items:
                xs.pop(j)
                try:
                    xs.append(next(it))
                except StopIteration:
                    i -= 1
                    cycle_length -= 1
            else:
                for item in items:
                    yield item
                i = j + 1


class Concat(Dataset):

    def __init__(self, a: Dataset, b: Dataset, name: str = None):
        super().__init__(name)
        assert isinstance(a, Dataset)
        assert isinstance(b, Dataset)
        self._datasets = (a, b)

    def __len__(self):
        return sum(map(len, self._datasets))

    def generator(self):
        for x in itertools.chain(*self._datasets):
            yield x


class Shard(Nested):
    def __init__(self, dataset: Dataset, num_shards: int, index: int, name: str = None):
        super().__init__(dataset, name)
        assert num_shards > 1
        assert index > -1
        assert index < num_shards
        shard_size, remain = divmod(len(dataset), num_shards)
        shard_sizes = [shard_size] * num_shards
        for i in range(0, min(index + 1, remain)):
            shard_sizes[i] += 1
        boundaries = [0] + list(itertools.accumulate(shard_sizes, lambda a, b: a + b)) + [len(dataset)]
        start, stop = boundaries[index], boundaries[index + 1]
        self._start = start
        self._stop = stop

    def __len__(self):
        return self._stop - self._start

    def generator(self):
        for x in itertools.islice(self._dataset, self._start, self._stop):
            yield x


class Shuffle(Nested):
    def __init__(self, dataset: Dataset, buffer_size: int, name: str = None):
        super().__init__(dataset, name)
        assert buffer_size > 1
        self._buffer_size = buffer_size

    def generator(self):
        it = iter(self._dataset)
        size = min(self._buffer_size, len(self._dataset))
        buffer = list(itertools.islice(it, size))
        indices = iter(numpy.random.randint(0, size, size))

        for x in it:
            try:
                i = next(indices)
            except StopIteration:
                indices = iter(numpy.random.randint(0, size, size))
                i = next(indices)
            yield buffer[i]
            buffer[i] = x

        numpy.random.shuffle(buffer)
        for x in buffer:
            yield x


class Sort(Nested):
    def __init__(self, dataset: Dataset, buffer_size: int, key: Callable[[Any], Any] = None, name: str = None):
        super().__init__(dataset, name)
        assert buffer_size > 1
        assert key is None or callable(key)
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


class Take(Nested):
    def __init__(self, dataset: Dataset, n: int, name: str = None):
        super().__init__(dataset, name)
        assert n > 0
        self._n = n

    def __len__(self):
        return self._n

    def generator(self):
        for x in itertools.islice(iter(self._dataset), self._n):
            yield x


class Skip(Dataset):
    def __init__(self, dataset: Dataset, n: int, name: str = None):
        super().__init__(name)
        assert n > -1
        self._n = n

    def __len__(self):
        return max(len(self._dataset) - self._n, 0)

    def generator(self):
        for x in itertools.islice(iter(self._dataset), self._n, len(self._dataset)):
            yield x


class Window(Nested):
    def __init__(self, dataset: Dataset, size: int, shift: int = None, stride: int = 1, drop_tail: bool = False,
                 name: str = None):
        super().__init__(dataset, name)
        assert size > 1
        if shift is None:
            shift = size
        assert shift > 0
        assert stride > 0
        self._size = size
        self._shift = shift
        self._stride = stride
        self._drop_tail = drop_tail

    def __len__(self):
        raise NotImplementedError

    def generator(self):
        raise NotImplementedError
