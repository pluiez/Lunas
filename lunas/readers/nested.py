from typing import List, Dict

import numpy
from lunas.readers.base import BaseReader, Reader, NestedReader
from overrides import overrides


class Shuffle(NestedReader):
    def __init__(self, reader: Reader, shuffle_size=None, buffer_size: int = 10000, num_threads: int = 1):
        """
        Apply random permutation to a given reader.
        Args:
            reader:
            shuffle_size: sequentially read samples from a reader into a fixed-sized buffer for permutation.
            buffer_size: process samples in buffer in parallel.
            num_threads: number of threads for parallel processing.
        """
        if shuffle_size < 0:
            shuffle_size = len(reader)
        shuffle_size = shuffle_size or buffer_size

        assert shuffle_size > 0, shuffle_size

        super().__init__(reader, buffer_size, num_threads)

        self._shuffle_size = shuffle_size
        self._shuffled_buffer = []
        self._random_state = numpy.random.get_state()

    def _shuffle(self):
        buffer = self._shuffled_buffer
        while len(buffer) < self._shuffle_size:
            try:
                buffer.append(super().next())
            except StopIteration:
                break
        numpy.random.shuffle(buffer)

    @overrides
    def next(self):
        if not self._shuffled_buffer:
            self._shuffle()
        try:
            return self._shuffled_buffer.pop(0)
        except IndexError:
            raise StopIteration

    @overrides
    def __iter__(self):
        self._random_state = numpy.random.get_state()
        return super().__iter__()

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        numpy.random.set_state(state_dict['_random_state'])
        del state_dict['_random_state']
        super().load_state_dict(state_dict)


class Zip(NestedReader):
    def __init__(self, reader: List[BaseReader], buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(reader, buffer_size, num_threads)
        sizes = list(map(len, reader))
        if len(set(sizes)) != 1:
            raise RuntimeError(
                f'Sizes of datasets {tuple(sizes)} must match.'
            )
        self._exclusions += ['reader']
