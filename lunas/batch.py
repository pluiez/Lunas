from typing import Callable, Any


class Batch(object):
    def __init__(self, capacity: int, sample_size_fn: Callable[[Any], int] = None, collate_fn=None):
        super().__init__()
        self._capacity = capacity  # max size
        self._sample_size_fn = sample_size_fn
        self._collate_fn = collate_fn

        self._max_sample_size = 0  # max sample size
        self._size = 0  # true size

        self._samples = []
        self._data = None
        self._num_samples = 0
        self._filled = False

    @property
    def capacity(self):
        return self._capacity

    @property
    def samples(self):
        return self._samples

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._max_sample_size * self.num_samples

    @property
    def num_samples(self):
        return len(self._samples)

    @property
    def filled(self):
        return self._filled

    def add(self, sample):
        if self.filled:
            return False
        self._samples.append(sample)
        self._num_samples += 1
        size = self._sample_size_fn(sample) if self._sample_size_fn else 1
        self._max_sample_size = max(size, self._max_sample_size)
        self._size = self._max_sample_size * self._num_samples
        self._filled = self.size >= self.capacity
        if self.filled:
            self._data = self._samples
            if self._collate_fn:
                self._data = self._collate_fn(self._samples)
        return not self.filled

    def pin_memory(self):
        try:
            from torch.utils.data._utils.pin_memory import pin_memory_batch
            self._data = pin_memory_batch(self.data)
        except ImportError as e:
            raise Exception('pin_memory requires PyTorch installation.') from e
        return self
