from __future__ import annotations

import abc
from collections import deque
from typing import List, Dict, Callable, Any

from lunas.persistable import Persistable
from lunas.utils import parallel_map, get_state_dict, load_state_dict


class BaseReader(Persistable):
    """A abstract class representing a dataset reader.

    A reader includes pipeline to process each sample and iterate through the dataset.
    This class defines any interface that's visible to users.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, bufsize: int = 10000, num_threads: int = 1):
        super().__init__()
        self._bufsize = bufsize
        self._num_threads = num_threads
        self._buffer: deque = deque()
        self._fns: List = []
        self._fast_skip: bool = False
        # Indicates position in the dataset.
        self._cursor: int = -1
        self._stop_iteration = False

        # Excludes these attributes from `self.state_dict()`
        self._inclusions: List[str] = ['_inclusions', '_fast_skip', '_cursor', '_stop_iteration']


    def size(self) -> int:
        """Get size of this dataset.

        Note: The returned value represents the total number of samples of the dataset, without filters
        applied to samples. When filters are applied, the effective size of dataset may be different
        from the value returned by this method. So please make sure the filters are applied to the final dataset
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

    def select(self, fn: Callable[[Any], Any]) -> BaseReader:
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
        self._fns.append(fn)
        return self

    def where(self, predicate: Callable[[Any], bool]) -> BaseReader:
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
        self._fns.append(predicate)
        return self

    def state_dict(self) -> Dict:
        return get_state_dict(self, inclusions=self._inclusions)

    def load_state_dict(self, state_dict: Dict) -> None:
        self._buffer.clear()
        load_state_dict(self, state_dict)

        cursor = self._cursor
        self._reset_cursor()
        # Fast-skip these samples
        is_fast_skip = self._fast_skip
        self._fast_skip = True

        while self._cursor < cursor:
            self._move_cursor()
            self._buffered_next()
        self._fast_skip = is_fast_skip
        if not is_fast_skip:
            self._process_buffer()

    def _move_cursor(self) -> int:
        """Increases the cursor and returns the new value.

        Returns:
            A `int` indicates current position.
        """
        self._cursor += 1
        return self._cursor

    def _reset_cursor(self):
        """Reset cursor to enable re-iterating over the dataset.

        """
        self._stop_iteration = False
        self._cursor: int = -1

    def _fill_buffer(self) -> int:
        size = 0
        if not self._stop_iteration:
            buffer = self._buffer
            for _ in range(self._bufsize - len(buffer)):
                try:
                    buffer.append(self.next())
                except StopIteration:
                    self._stop_iteration = True
                    break
            size = len(buffer)
            if not self._fast_skip and size > 0:
                self._process_buffer()
        return size

    def _process_buffer(self, buffer=None, inplace=True):
        buffer = self._buffer if buffer is None else buffer
        buffer = self._parallel_apply(buffer)
        # buffer = list(filter(None, buffer))

        if inplace:
            self._buffer = deque(buffer)
        return buffer

    def _buffered_next(self) -> Any:
        """Wrapper method to get next sample.

        Wraps `self.next()`, apply transformations and predicates, and maintains
        internal state of this object.

        Returns:
            A sample of any type.

        """
        try:
            return self._buffer.popleft()
        except IndexError:
            size = self._fill_buffer()
            if size == 0:
                raise StopIteration
            return self._buffer.popleft()

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

        for fn in self._fns:
            new_sample = fn(sample)

            # Stops when predicate is evaluated to False.
            if new_sample is False:
                return None
            elif new_sample is not True:
                sample = new_sample

        return sample

    def _star_apply(self, sample: Any) -> Any:
        if sample is None:
            return None

        for fn in self._fns:
            try:
                new_sample = fn(*sample)
            except TypeError as e:
                raise Exception(f'When a reader returns a tuple or list as a sample, '
                                f'the mapping functions (`Reader.select()` and `Reader.where()`) '
                                f'for succeeding readers should take equivalent number of parameters as args. '
                                f'Please check the mapping functions for `{type(self)}`') from e

            # Stops when predicate is evaluated to False.
            if new_sample is False:
                return None
            elif new_sample is not True:
                sample = new_sample

        return sample

    def _parallel_apply(self, samples: List[Any]) -> List[Any]:
        if self._fns and samples:
            if not isinstance(samples, (tuple, list)):
                samples = list(samples)
            if isinstance(samples[0], (tuple, list)):
                return parallel_map(self._star_apply, samples, self._num_threads)
            else:
                return parallel_map(self._apply, samples, self._num_threads)
        else:
            return samples

    def _finalize(self):
        """Release resources after iteration stops.

        """
        pass

    def __next__(self):
        """Implements iterator method and returns a sample.

        Returns:
            A sample instance.
        """
        try:
            sample = self._buffered_next()
            self._move_cursor()
        except StopIteration as e:
            self._finalize()
            raise e

        if sample is None:
            return self.__next__()
        return sample

    def __iter__(self) -> BaseReader:
        """Returns an iterator of this dataset.

        """
        self._reset_cursor()
        return self

    def __len__(self):
        """Returns size of this dataset.

        Returns:
            A `int` scalar.
        """
        return self.size()
