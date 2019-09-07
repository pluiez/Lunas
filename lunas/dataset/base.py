from __future__ import annotations

import abc
import functools
import inspect
import itertools
from collections import deque
from typing import List, Dict, Callable, Any

from lunas.persistable import Persistable
from lunas.utils import get_state_dict, load_state_dict


class BaseDataset(Persistable):
    """A abstract class representing a dataset.

    A dataset includes pipeline to process each sample and iterate through the dataset.
    This class defines any interface that's visible to users.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, bufsize: int = 1):
        super().__init__()
        assert bufsize > 0, bufsize

        self._bufsize = bufsize
        self._buffer: deque = deque()
        self._fns: List = []
        # Indicates position in the dataset.
        self._buf_counter: int = 0
        self._buf_ptr = 0
        self._random_state = None
        # Includes these attributes from `self.state_dict()`
        self._initialized = False
        self._buffer_callbacks = []

        self._enable_processing = True

        self._inclusions: List[str] = ['_inclusions', '_buf_ptr', '_buf_counter', '_random_state', '_bufsize']

    @property
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

    def select(self, fn: Callable[[Any], Any]) -> BaseDataset:
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

    def where(self, predicate: Callable[[Any], bool]) -> BaseDataset:
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
        load_state_dict(self, state_dict)

        self.initialize()

        self.enable_processing(False)
        for _ in range(0, state_dict['_buf_counter'] - 1):
            self.fill_buffer(False)

        self.enable_processing(True)
        self.fill_buffer_(True)
        i = 0
        while i < state_dict['_buf_ptr']:
            self.__next__()
            i += 1
        self.process_()

    def process(self, buffer):
        if self._fns and buffer:
            buffer = iter(buffer)
            sample = next(buffer)  # iterable

            apply = functools.partial(self._apply,
                                      unpack_list=isinstance(sample, (tuple, list)),
                                      unpack_dict=False)
            sample = apply(sample)
            buffer = map(apply, buffer)
            buffer = itertools.chain([sample], buffer)

        return buffer

    def process_(self, buffer=None, cast_to=deque):
        buffer = self.process(buffer or self._buffer)
        if cast_to is not None and buffer is not None and not isinstance(buffer, cast_to):
            buffer = cast_to(buffer)
        self._buffer = buffer

    def _apply(self, sample: Any, unpack_list=False, unpack_dict=False) -> Any:
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
            try:
                if unpack_list:
                    new_sample = fn(*sample)
                elif unpack_dict:
                    new_sample = fn(**sample)
                else:
                    new_sample = fn(sample)
            except TypeError as e:
                raise Exception(f'Incompatible mapping function and inputs in `{type(self)}`.'
                                f'Function signature: {inspect.signature(fn)}.'
                                f'Inputs: {sample}.') from e

            # Stops when predicate is evaluated to False.
            if new_sample is False:
                return None
            elif new_sample is not True:
                sample = new_sample
        return sample

    def enable_processing(self, enable: bool):
        self._enable_processing = enable

    def initialize(self):
        self._initialized = True
        self._buf_counter, self._buf_ptr = 0, 0
        self._buffer.clear()
        self.enable_processing(True)

    def finalize(self):
        """Release resources after iteration stops.

        """
        self._initialized = False
        pass

    def fill_buffer(self, callback=True):
        buffer = []
        for _ in range(self._bufsize):
            try:
                buffer.append(self.next())
            except StopIteration:
                break
        if not buffer:
            self.finalize()
            raise StopIteration

        self._buf_counter += 1

        if callback:
            for func in self._buffer_callbacks:
                buffer = func(buffer)
        return buffer

    def fill_buffer_(self, callback=True):
        self._buffer = deque(self.fill_buffer(callback))

    def _buffered_next(self):
        # self._lock.acquire()
        if not self._buffer:
            buffer = self.fill_buffer()
            self._buffer = deque(buffer)
            # Skip processing

        x = self._buffer.popleft()
        self._buf_ptr += 1
        # self._lock.release()
        return x

    def __next__(self):
        if not self._buffer:
            buffer = self.fill_buffer()
            self._buf_ptr = 0
            self.process_(buffer=buffer)

        x = self._buffer.popleft()  # FIFO
        self._buf_ptr += 1
        if x is not None:
            return x
        return self.__next__()

    def __iter__(self) -> BaseDataset:
        """Returns an iterator of this dataset.

        """
        if self._initialized:
            self.finalize()
        self.initialize()

        return self

    def __len__(self):
        """Returns size of this dataset.

        Returns:
            A `int` scalar.
        """
        return self.size
