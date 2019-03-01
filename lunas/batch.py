from collections import deque
from typing import List, Iterable, Dict, Callable, Any

import numpy
from overrides import overrides

from lunas.persistable import Persistable
from lunas.utils import get_state_dict, load_state_dict


class Batch(Persistable):
    def __init__(self, max_size: int, size_fn: Callable[[Any], int] = None,
                 max_padded_size: float = None, padded_size_fn: Callable[[List[Any]], int] = None):
        """
        A sizable data container for holding a batch of samples.
        Args:
            max_size: Maximum batch size.
            size_fn: A callable function that returns the size of a given sample.
            max_padded_size: (Optional.) Maximum size that consider paddings in the resulting tensor.
            padded_size_fn: A callable function that returns the padded size of given samples.
        """
        super().__init__()
        self._max_size = max_size
        self._size_fn = size_fn
        self._max_padded_size = max_padded_size
        self._padded_size_fn = padded_size_fn

        self._samples: deque = deque()
        self._effective_size: int = 0
        self._sort_idx: numpy.ndarray = None
        self._exclusions = ['_size_fn']
        self._data = None

        self._filled = True

    @property
    def data(self):
        return self._data

    @property
    def samples(self):
        return [sample for sample, size in self._samples]

    @property
    def sizes(self):
        return [size for sample, size in self._samples]

    @property
    def effective_size(self):
        return self._effective_size

    @property
    def filled(self):
        return self._filled

    @property
    def num_samples(self):
        return len(self._samples)

    def pop_all(self):
        samples = self.samples
        self._samples.clear()
        self._effective_size = 0
        return samples

    def pop(self, fifo=True):
        if fifo:
            sample, size = self._samples.popleft()
        else:
            sample, size = self._samples.pop()
        self._effective_size -= size
        return sample

    def push(self, sample):
        size = 1
        if self._size_fn is not None:
            size = self._size_fn(sample)
        sample = (sample, size)
        self._samples.append(sample)
        self._effective_size += size

    def strip(self, max_size=None):
        max_size = max_size or self._max_size
        rv = []
        while self.effective_size > max_size:
            rv.append(self.pop(False))

        return rv[::-1]

    def from_iter(self, sample_iter, size: int = None, raise_when_stopped: bool = False):
        """
        Fills batch from an iterable object. Enables dynamic batch size if size is not None.
        Args:
            sample_iter:
            size:
            raise_when_stopped:
        Returns:
            added: int, size of added samples.
        """
        size = size or self._max_size
        # Approximate
        init_size = self.effective_size
        self._filled = True
        while self.effective_size < size and (
                not self._max_padded_size or
                self._padded_size_fn(self.samples) < self._max_padded_size
        ):
            try:
                self.push(next(sample_iter))
            except StopIteration as e:
                self._filled = False
                if raise_when_stopped:
                    raise e
                else:
                    break
        return self.effective_size - init_size

    def from_deque(self, sample_list: deque, size: int = None):
        size = size or self._max_size
        init_size = self.effective_size
        self._filled = True
        while self.effective_size < size and (
                not self._max_padded_size or
                self._padded_size_fn(self.samples) < self._max_padded_size
        ):
            try:
                self.push(sample_list.popleft())
            except IndexError:
                self._filled = False
                break
        return self.effective_size - init_size

    def sort(self, key_fn: Callable[[Any], int]):
        if key_fn is not None:
            keys = numpy.array(list(map(key_fn, self.samples)))
            indices = numpy.argsort(keys)
            self._sort_idx = indices
            samples = list(self._samples)  # convert to list for faster indexing
            self._samples = deque([samples[i] for i in indices])

    def revert(self, samples: List[Any] = None):
        if self._sort_idx is None:
            return samples
        else:
            indices = numpy.argsort(self._sort_idx)
            indices = list(indices)

        arg_not_none = samples is not None

        if samples is None:
            samples = list(self._samples)
        else:
            if len(samples) != len(self._sort_idx):
                raise RuntimeError(
                    f'The number of samples ({len(samples)}) must match '
                    f'the number of indices ({len(self._sort_idx)}).'
                )
        samples = [samples[i] for i in indices]

        if arg_not_none:
            return samples
        else:
            self._samples = deque(samples)

    def process(self, collate_fn):
        self._data = collate_fn(self.samples)

    @overrides
    def state_dict(self) -> Dict:
        return get_state_dict(self, exclusions=self._exclusions)

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        load_state_dict(self, state_dict)


class Cache(Batch, Iterable):
    def __init__(self, max_size: int, size_fn: Callable[[Any], int] = None):
        super().__init__(max_size, size_fn)

    def __next__(self):
        if self._samples:
            return self.pop()
        raise StopIteration

    def __iter__(self):
        return self
