from typing import List, Dict

import numpy
from overrides import overrides

from lunas.readers.base import BaseReader, Reader, NestedReader


class Shuffle(NestedReader):
    def __init__(self, reader: Reader, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(reader, buffer_size, num_threads)
        self._random_state = numpy.random.get_state()

    def _shuffle_buffer(self):
        numpy.random.shuffle(self._buffer)

    @overrides
    def _fill_buffer(self):
        rv = super()._fill_buffer()
        self._shuffle_buffer()
        return rv

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
