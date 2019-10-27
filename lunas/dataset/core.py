from __future__ import annotations

import abc
import inspect
import itertools
import sys
from typing import List, Callable, Any, Union, Tuple

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
        self._ref_count = 0

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def generator(self):
        raise NotImplementedError

    def __iter__(self):
        ptr = self._ptr % len(self)
        for self._ptr, x in enumerate(itertools.islice(self.generator(), ptr, None), ptr + 1):
            x = self._apply(x)
            if x is not None:
                yield x

    @property
    def name(self):
        return self.name

    def state_dict(self) -> dict:
        return {'ptr': self._ptr}

    def load_state_dict(self, state: dict) -> None:
        self._ptr = state['ptr']

    def chk_and_inc_ref(self) -> None:
        if self._ref_count > 0:
            raise RuntimeError(f'Dataset `{self}` is only allowed to be wrapped in another dataset once, '
                               'you should consider create a new instance instead.')
        else:
            self._ref_count += 1

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

    def reset(self) -> None:
        self._ptr = 0

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

    def slice(self, start=None, stop=None, name: str = None) -> Slice:
        return Slice(self, start, stop, name)

    def take(self, n: int, name: str = None) -> Slice:
        return Slice(self, None, n, name)

    def skip(self, n: int, name: str = None) -> Slice:
        return Slice(self, n, None, name)

    def window(self, size: int, shift: int = None, stride: int = 1,
               drop_tail: bool = False, name: str = None) -> Window:
        return Window(self, size, shift, stride, drop_tail, name)


class NestedN(Dataset):

    def __init__(self, datasets: Union[Tuple[Dataset], List[Dataset]], name: str = None):
        super().__init__(name)
        assert isinstance(datasets, (tuple, list)), type(datasets)
        datasets = tuple(datasets)
        assert all(map(lambda _: isinstance(_, Dataset),
                       datasets)), f'datasets must subclass Dataset. ' \
                                   f'Got: {tuple(map(lambda _: isinstance(_, Dataset), datasets))}'
        for x in datasets:
            x.chk_and_inc_ref()
        self._datasets = datasets

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def generator(self):
        raise NotImplementedError

    def __iter__(self):
        self._ptr = self._ptr % len(self)
        if self._ptr == 0:
            self.reset()
        for self._ptr, x in enumerate(self.generator(), self._ptr + 1):
            x = self._apply(x)
            if x is not None:
                yield x

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({'datasets': [x.state_dict() for x in self._datasets]})
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        for x, state in zip(self._datasets, state['datasets']):
            x.load_state_dict(state)

    def reset(self) -> None:
        super().reset()
        for x in self._datasets:
            x.reset()


class Nested(NestedN):

    def __init__(self, dataset: Dataset, name: str = None):
        super().__init__([dataset], name)
        self._dataset = dataset

    @abc.abstractmethod
    def generator(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._dataset)


class Repeat(Nested):

    def __init__(self, dataset: Dataset, n: int = None, name: str = None):
        super().__init__(dataset, name)
        assert n is None or (n > 0 and isinstance(n, int))
        self._n = n

    def __len__(self):
        return len(self._dataset) * self._n if self._n is not None else sys.maxsize

    def generator(self):
        repeats = itertools.repeat(self._dataset, self._n) if self._n else itertools.repeat(self._dataset)
        for x in itertools.chain.from_iterable(repeats):
            yield x


class InterleaveDataset(Nested):

    def __init__(self, dataset: Dataset, map_fn: Callable[[Any], Dataset], cycle_length: int, block_length: int,
                 name: str = None):
        super().__init__(dataset.select(lambda x: iter(map_fn(x))), name)
        assert callable(map_fn)
        assert cycle_length > 0
        assert block_length > 0
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
    def __init__(self, dataset: Dataset, buffer_size: int, name: str = None):
        super().__init__(dataset, name)
        assert buffer_size > 1
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


class Slice(Nested):

    def __init__(self, dataset: Dataset, start: int = None, stop: int = None, name: str = None):
        super().__init__(dataset, name)
        assert not (start is None and stop is None)

        if start is None:
            assert stop > 0
            size = min(stop, len(dataset))
        else:
            assert start > -1
            if stop is None:
                size = len(dataset) - start
            else:
                size = min(stop, len(dataset)) - start
            assert size > 0, size
        self._skipped = False
        self._size = size
        self._start = start
        self._stop = stop

    def __len__(self):
        return self._size

    def generator(self):
        if self._start is None:
            it = itertools.islice(self._dataset, self._stop)
        else:
            # ONLY itertools.islice(iterable, stop) is capable of preserving iteration state.
            # it = itertools.islice(self._dataset, self._start, self._stop)
            if not self._skipped:
                for _ in itertools.islice(self._dataset, self._start):
                    ...
                self._skipped = True
            it = itertools.islice(self._dataset, len(self) - self._ptr)
        for x in it:
            yield x

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({'skipped': self._skipped})
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._skipped = state['skipped']

    def reset(self) -> None:
        super().reset()
        self._skipped = False


class Shard(Slice):
    def __init__(self, dataset: Dataset, num_shards: int, index: int, name: str = None):
        assert num_shards > 1
        assert index > -1
        assert index < num_shards
        shard_size, remain = divmod(len(dataset), num_shards)
        shard_sizes = [shard_size] * num_shards
        for i in range(0, min(index + 1, remain)):
            shard_sizes[i] += 1
        boundaries = [0] + list(itertools.accumulate(shard_sizes, lambda a, b: a + b)) + [len(dataset)]
        start, stop = boundaries[index], boundaries[index + 1]
        super().__init__(dataset, start, stop, name)


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
