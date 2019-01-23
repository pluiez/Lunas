import abc
import itertools
from typing import List, Dict, Callable, Any

from overrides import overrides

from lunas.persistable import Persistable
from lunas.utils import parallel_map, get_state_dict, load_state_dict


class BaseReader(Persistable):
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
        raise NotImplementedError

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
        ds1 = Range(10).where(lambda x: x<5)
        ds2 = Range(10).where(lambda x: x<6)
        ds = Zip(ds1, ds2)
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
        outside, use `next(dataset)` instead. This method should raise StopIteration at the end of dataset.

        Returns:
            A sample of any type.
        """
        raise NotImplementedError

    def buffered_next(self) -> Any:
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

    def load_state_dict(self, state_dict: Dict) -> None:
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError


class Reader(BaseReader):
    """Implements the functions defined in `BaseReader`.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__()
        self._buffer_size = buffer_size
        self._num_threads = num_threads
        self._buffer: List = []
        self._fns: List = []
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

    def get_effective_buffer_size(self) -> int:
        return len(self._buffer)

    def process_buffer(self, buffer=None, inplace=True):
        buffer = self._buffer if buffer is None else buffer
        buffer = self._parallel_apply(buffer)
        # buffer = list(filter(None, buffer))

        if inplace:
            self._buffer = buffer
        return buffer

    def _fill_buffer(self) -> int:
        buffer = self._buffer
        while self.get_effective_buffer_size() < self._buffer_size:
            try:
                buffer.append(self.next())
            except StopIteration:
                break
        size = len(buffer)
        if not self._fast_skip and size > 0:
            self.process_buffer()
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
        if sample is None:
            return sample

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
    def buffered_next(self) -> Any:
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
        self._buffer.clear()
        load_state_dict(self, state_dict)

        cursor = self._cursor
        self.reset_cursor()
        # Fast-skip these samples
        is_fast_skip = self._fast_skip
        self._fast_skip = True

        while self._cursor < cursor:
            self.cursor()
            self.buffered_next()
        self._fast_skip = is_fast_skip
        if not is_fast_skip:
            self.process_buffer()

    @overrides
    def finalize(self):
        pass

    @overrides
    def __next__(self) -> Any:
        try:
            sample = self.buffered_next()
            self.cursor()
        except StopIteration as e:
            self.finalize()
            raise e

        if sample is None:
            return self.__next__()
        return sample

    def __iter__(self) -> BaseReader:
        return super().__iter__()


class NestedReader(Reader):
    def __init__(self, reader, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(buffer_size, num_threads)
        multi_reader = isinstance(reader, (list, tuple))
        if not multi_reader:
            reader = [reader]
        for r in reader:
            r._fast_skip = True
        self.reader = reader
        self._multi_reader = multi_reader

    @overrides
    def size(self) -> int:
        return len(self.reader[0])

    @overrides
    def next(self) -> Any:
        sample = tuple([r.buffered_next() for r in self.reader])
        return sample if self._multi_reader else sample[0]

    def finalize(self):
        for reader in self.reader:
            reader.finalize()
        super().finalize()

    @overrides
    def process_buffer(self, buffer=None, inplace=True):
        buffer = self._buffer if buffer is None else buffer
        reader = self.reader
        if self._multi_reader:
            new_buffer = list(zip(
                *[r.process_buffer(buf, inplace=False) for r, buf in
                  zip(reader, itertools.zip_longest(*buffer))]))
            new_buffer = list(map(
                lambda sample: sample if all(s is not None for s in sample) else None,
                new_buffer
            ))
        else:
            new_buffer = reader[0].process_buffer(buffer, inplace=False)

        return super().process_buffer(new_buffer, inplace)

    @overrides
    def __iter__(self):
        self.reader = list(map(iter, self.reader))
        return super().__iter__()

    @overrides
    def state_dict(self) -> Dict:
        state = super().state_dict()
        state['reader'] = [r.state_dict() for r in self.reader]
        return state

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        for r, state in zip(self.reader, state_dict['reader']):
            r.load_state_dict(state)
        del state_dict['reader']

        super().load_state_dict(state_dict)
