from typing import List, Iterable, Dict, Callable, Any

import numpy


class Batch(object):
    def __init__(self, size: int, size_fn: Callable[[Any], int]):
        super().__init__()
        self.size = size
        self.size_fn = size_fn
        self._samples = []
        self._sort_idx: numpy.ndarray = None
        self._sample_sizes: Dict[int, int] = dict()

    @property
    def samples(self):
        return self._samples

    def pop_all(self):
        samples = self._samples
        self._samples = []
        self._sample_sizes = dict()
        return samples

    def pop(self, idx: int = -1):
        sample = self._samples.pop(idx)
        if id(sample) in self._sample_sizes:
            del self._sample_sizes[id(sample)]
        return sample

    def push(self, sample):
        self._samples.append(sample)
        self._sample_sizes[id(sample)] = self.size_fn(sample)

    def get_sample_sizes(self) -> Iterable:
        return (self._sample_sizes[id(x)] for x in self._samples)

    def effective_size(self):
        return sum(self.get_sample_sizes())

    def from_iter(self, sample_iter: Iterable, size: int = None, raise_when_stopped: bool = False):
        """
        Fills batch from an iterable object. Enables dynamic batch size if size is not None.
        Args:
            sample_iter:
            size:
            raise_when_stopped:

        """
        size = size or self.size
        # Approximate
        while self.effective_size() < size:
            try:
                self.push(next(sample_iter))
            except StopIteration as e:
                if raise_when_stopped:
                    raise e
                else:
                    break

    def from_list(self, sample_list: List, size: int = None):
        size = size or self.size
        while self.effective_size() < size:
            try:
                self.push(sample_list.pop(0))
            except IndexError:
                break

    def strip_batch_by_size(self, size: int = None):
        """
        Ensure effective batch size is limited.
        Args:
            size:

        Returns:

        """
        size = size or self.size
        rv = []
        while self.effective_size() > size and len(self._samples) > 1:
            rv.append(self.pop(-1))
        return rv

    def sort(self, key_fn: Callable[[Any], int]):
        if key_fn is not None:
            keys = numpy.array(list(map(key_fn, self._samples)))
            indices = numpy.argsort(keys)[::-1]
            self._sort_idx = indices
            indices = list(indices)
            self._samples = [self._samples[i] for i in indices]

    def desort(self, samples: List[Any]):
        if len(samples) != len(self._sort_idx):
            raise RuntimeError(f'Number of samples must match the number of indices. '
                               f'{len(samples)} != {len(self._sort_idx)}')
        indices = numpy.argsort(self._sort_idx)
        indices = list(indices)
        samples = [samples[i] for i in indices]
        return samples


class Cache(Batch, Iterable):
    def __init__(self, size: int, size_fn: Callable[[Any], int]):
        super().__init__(size, size_fn)

    def __next__(self):
        if self._samples:
            return self.pop(0)
        raise StopIteration

    def __iter__(self):
        return self
