import abc
from typing import List, Iterable, Dict, Callable, Any

import numpy
from overrides import overrides

from lunas.interface import Persistable
from lunas.utils import parallel_map, get_state_dict, load_state_dict


class BaseReader(Iterable, Persistable):
    """A abstract class representing a dataset reader.

    A reader includes pipeline to process each sample and iterate through the dataset.
    This class defines any interface that's visible to users.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()
        # Excludes these attributes from `self.state_dict()`
        self._exclusions: List[str] = []

    def cursor(self) -> int:
        """Increases the cursor and returns the new value.

        Returns:
            A `int` indicates current position.
        """
        raise NotImplementedError

    def reset_cursor(self) -> None:
        """Reset cursor to enable re-iterating over the dataset.

        """

    def where(self, predicate: Callable[[Any], bool]):
        """Filters a sample by predicate.

        A predicate is a callable object that takes a sample as input and returns a
        `bool` value to indicate whether this sample should be filtered out.

        Note that the predicate is not applied to the dataset immediately. Instead, we
        apply predicates to a large buffer and utilizes multi-threading pool to speed up data
        pipeline.

        Args:
            predicate: A `Callable` object.

        Returns:
            `self`.
        """
        raise NotImplementedError

    def select(self, fn: Callable[[Any], Any]):
        """Transforms a sample.

        A fn is a callable object that takes a sample as input and returns a transformed
        sample.

        Note that the transformation is not applied to the dataset immediately. Instead, we
        apply transformations to a large buffer and utilizes multi-threading pool to speed up data
        pipeline.

        Args:
            fn: A `Callable` object.

        Returns:
            `self`.

        """
        raise NotImplementedError

    def size(self) -> int:
        """Get size of this dataset.

        Note: The returned value represents the total number of samples of the dataset, without filters
        applied to samples. When filters are applied, the effective size of this dataset may be different
        the value returned by this method. So please make sure the filters are applied to the final dataset
        wrapper to avoid size inconsistency.

        For example, the following usage of `self.where()` filters would cause size inconsistency.
        ```python
        ds1 = RangeReader(10).where(lambda x: x<5)
        ds2 = RangeReader(10).where(lambda x: x<6)
        ds = ZipReader(ds1, ds2)
        iterator = DataIterator(ds)
        # ...
        for batch in iterator():
            print(batch)
        ```

        The effective size of `ds1` at runtime is 5, while it's 6 for `ds2`. This inconsistency cannot be checked
        before iterating through `ds`.

        Returns:
            A scalar of type `int`.

        """
        raise NotImplementedError

    def next(self) -> Any:
        """Get an original sample from the dataset by index. DO NOT invoke this method from
        outside, use `next(dataset)` instead.

        Returns:
            A sample of any type.
        """
        raise NotImplementedError

    def _next(self) -> Any:
        """Wrapper method to get next sample.

        Wraps `self.next()`, apply transformations and predicates, and maintains
        internal state of this object.

        Returns:
            A sample of any type.

        """
        raise NotImplementedError

    def _apply(self, sample: Any) -> Any:
        """Applies transformations and filters to a sample.

        When sample is filtered out, returns None, else returns transformed sample.

        Args:
            sample: An input sample to be processed.

        Returns:
            A sample or `None` if it's filtered.
        """
        raise NotImplementedError

    def finalize(self):
        """Release resources after iteration stops.

        """
        raise NotImplementedError

    def __len__(self):
        """Returns size of this dataset.

        Returns:
            A `int` scalar.
        """
        return self.size()

    def __next__(self) -> Any:
        """Implements iterator method and returns a sample.

        Returns:
            A sample instance.
        """
        raise NotImplementedError

    def __iter__(self):
        """Returns an iterator of this dataset.

        """
        self.reset_cursor()
        return self

    def shuffle(self, buffer_size: int = 10000, num_threads: int = 1):
        """Returns a shufflable counterpart of this dataset.

        Args:
            buffer_size: A `int` scalar. Represents the number of samples to shuffle each time.
            num_threads:

        Returns:
            A `ShuffleReader`.

        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict) -> None:
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError


class Reader(BaseReader):
    """Implements the functions defined in `BaseReader`.

    """

    def __init__(self, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__()
        self._buffer_size = buffer_size
        self._num_threads = num_threads
        self._buffer: List[Any] = []
        self._fns = []
        self._fast_skip: bool = False
        # Indicates position in the dataset.
        self._cursor: int = -1

        self._exclusions += ['_fns', '_buffer']

    @overrides
    def cursor(self) -> int:
        self._cursor += 1
        return self._cursor

    @overrides
    def reset_cursor(self):
        self._cursor: int = -1

    def check_buffer_size(self) -> None:
        """Evaluates the buffer size lazily to prevent ``self.size()`` function from invoking before
        sub-class is properly configured.

        """
        buffer_size = self._buffer_size
        buffer_size = buffer_size if buffer_size > 0 else len(self)
        self._buffer_size = min(buffer_size, len(self))

    def get_effective_buffer_size(self) -> int:
        return len(self._buffer)

    def _fill_buffer(self) -> int:
        self.check_buffer_size()
        buffer = self._buffer
        while self.get_effective_buffer_size() < self._buffer_size:
            try:
                buffer.append(self.next())
            except StopIteration:
                break
        size = len(buffer)
        if not self._fast_skip and size > 0:
            self._buffer = self._parallel_apply(self._buffer)
        return size

    @overrides
    def select(self, fn: Callable[[Any], Any]) -> BaseReader:
        self._fns.append(fn)
        return self

    @overrides
    def where(self, predicate: Callable[[Any], bool]) -> BaseReader:
        self._fns.append(predicate)
        return self

    @overrides
    def _apply(self, sample: Any) -> Any:
        for fn in self._fns:
            new_sample = fn(sample)
            # Stops when predicate is evaluated to False.
            if new_sample is False:
                return None
            elif new_sample is not True:
                sample = new_sample

        return sample

    def _parallel_apply(self, samples: List[Any]) -> List[Any]:
        if self._fns:
            return parallel_map(self._apply, samples, self._num_threads)
        else:
            return samples

    @overrides
    def _next(self) -> Any:

        if self.get_effective_buffer_size() == 0:
            self._fill_buffer()
        try:
            return self._buffer.pop(0)
        except IndexError:
            raise StopIteration

    @overrides
    def state_dict(self) -> Dict:
        return get_state_dict(self, exclusions=self._exclusions)

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        load_state_dict(self, state_dict)
        cursor = self._cursor
        self.reset_cursor()
        # Fast-skip these samples
        self._fast_skip = True
        while self._cursor < cursor:
            self.cursor()
            self._next()
        self._buffer = self._parallel_apply(self._buffer)
        self._fast_skip = False

    @overrides
    def finalize(self):
        pass

    @overrides
    def __next__(self) -> Any:
        # Check whether reaches the end of the dataset.
        if self._cursor + 1 >= self.size():
            self.finalize()
            raise StopIteration

        self.cursor()
        sample = self._next()
        # Don't return filtered samples.
        if sample is None:
            return self.__next__()
        return sample

    def __iter__(self) -> BaseReader:
        return super().__iter__()

    def shuffle(self, buffer_size: int = 10000, num_threads: int = 1) -> BaseReader:
        return ShuffleReader(self, buffer_size, num_threads)


class ShuffleReader(Reader):
    def __init__(self, reader: BaseReader, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(buffer_size, num_threads)
        self.reader = reader
        self._random_state = numpy.random.get_state()

    @overrides
    def size(self) -> int:
        return len(self.reader)

    def next(self) -> Any:
        rv = self.reader._next()
        return rv

    def _shuffle_buffer(self):
        numpy.random.shuffle(self._buffer)

    @overrides
    def _fill_buffer(self):
        rv = super()._fill_buffer()
        self._shuffle_buffer()
        return rv

    def finalize(self):
        self.reader.finalize()
        super().finalize()

    @overrides
    def __iter__(self) -> Iterable:
        self._random_state = numpy.random.get_state()
        self.reader = iter(self.reader)
        return super().__iter__()

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        numpy.random.set_state(state_dict['_random_state'])
        del state_dict['_random_state']
        super().load_state_dict(state_dict)


class ZipReader(Reader):
    def __init__(self, *readers: List[BaseReader], buffer_size: int = 10000, num_threads: int = 1,
                 check_size_consistency: bool = True):
        super().__init__(buffer_size, num_threads)
        self.readers = readers
        if check_size_consistency:
            sizes = list(map(len, readers))
            if len(set(sizes)) != 1:
                raise RuntimeError(f'Sizes of datasets must match. Got {sizes}.')
        self._exclusions += ['readers']

    @overrides
    def size(self) -> int:
        return len(self.readers[0])

    @overrides
    def next(self):
        sample = tuple([r._next() for r in self.readers])
        # sample = tuple(r.next() for r in self.readers)
        if any(s is None for s in sample):
            sample = None

        return sample

    def finalize(self):
        for reader in self.readers:
            reader.finalize()
        super().finalize()

    @overrides
    def __iter__(self) -> Iterable:
        self.readers = list(map(iter, self.readers))
        return super().__iter__()

    @overrides
    def state_dict(self) -> Dict:
        state = super().state_dict()
        state['readers'] = [r.state_dict() for r in self.readers]
        return state

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        for r, state in zip(self.readers, state_dict['readers']):
            r.load_state_dict(state)
        del state_dict['readers']

        super().load_state_dict(state_dict)


class RangeReader(Reader):
    def __init__(self, start: int, stop: int = None, step: int = None, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(buffer_size, num_threads)
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1

        self.start = start
        self.stop = stop
        self.step = step

        self._range = None

        self._exclusions += ['_range']

    @overrides
    def size(self) -> int:
        import math
        return int(math.ceil((self.stop - self.start) / self.step))

    def _reset(self):
        start, stop, step = self.start, self.stop, self.step
        self._range = iter(range(start, stop, step))

    @overrides
    def reset_cursor(self):
        super().reset_cursor()
        self._reset()

    @overrides
    def next(self) -> Any:
        return next(self._range)
