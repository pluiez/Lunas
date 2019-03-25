import itertools
from collections import deque
from typing import List, Dict

import numpy
from lunas.readers.base import BaseReader
from overrides import overrides


class Nested(BaseReader):
    def __init__(self, reader, bufsize: int = 10000, num_threads: int = 1):
        super().__init__(bufsize, num_threads)
        multi_reader = isinstance(reader, (list, tuple))
        if not multi_reader:
            reader = [reader]
        for r in reader:
            r._fast_skip = True
        self.reader = reader
        self._multi_reader = multi_reader
        self._inclusions += ['reader', '_multi_reader']

    @overrides
    def size(self) -> int:
        return len(self.reader[0])

    @overrides
    def next(self):
        sample = tuple([r._buffered_next() for r in self.reader])
        return sample if self._multi_reader else sample[0]

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

    @overrides
    def _process_buffer(self, buffer=None, inplace=True):
        buffer = self._buffer if buffer is None else buffer
        reader = self.reader
        if self._multi_reader:
            new_buffer = list(zip(
                *[r._process_buffer(buf, inplace=False) for r, buf in
                  zip(reader, itertools.zip_longest(*buffer))]))
            new_buffer = list(map(
                lambda sample: sample if all(s is not None for s in sample) else None,
                new_buffer
            ))
        else:
            new_buffer = reader[0]._process_buffer(buffer, inplace=False)

        return super()._process_buffer(new_buffer, inplace)

    @overrides
    def _finalize(self):
        for reader in self.reader:
            reader._finalize()
        super()._finalize()

    @overrides
    def __iter__(self):
        self.reader = list(map(iter, self.reader))
        return super().__iter__()


class Shuffle(Nested):
    def __init__(self, reader: BaseReader, shufsize=None, bufsize: int = 10000, num_threads: int = 1):
        """
        Apply random permutation to a given reader.
        Args:
            reader:
            shufsize: sequentially read samples from a reader into a fixed-sized buffer for permutation.
            bufsize: process samples in buffer in parallel.
            num_threads: number of threads for parallel processing.
        """
        if not shufsize or shufsize < 0:
            shufsize = len(reader)
        shufsize = shufsize or bufsize

        assert shufsize > 0, shufsize

        super().__init__(reader, bufsize, num_threads)

        self._shufsize = shufsize
        self._shuffled_buffer: deque = deque()
        self._random_state = numpy.random.get_state()

        self._inclusions += ['_random_state']

    @overrides
    def next(self):
        if not self._shuffled_buffer:
            self._shuffle()
        try:
            sample = self._shuffled_buffer.popleft()
        except IndexError:
            raise StopIteration
        return sample

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        numpy.random.set_state(state_dict['_random_state'])
        del state_dict['_random_state']
        super().load_state_dict(state_dict)

    def _shuffle(self):
        buffer = self._shuffled_buffer
        while len(buffer) < self._shufsize:
            try:
                buffer.append(super().next())
            except StopIteration:
                break
        buffer = list(buffer)
        numpy.random.shuffle(buffer)
        self._shuffled_buffer = deque(buffer)

    @overrides
    def __iter__(self):
        self._random_state = numpy.random.get_state()
        return super().__iter__()


class Zip(Nested):
    def __init__(self, reader: List[BaseReader], bufsize: int = 10000, num_threads: int = 1):
        super().__init__(reader, bufsize, num_threads)
        sizes = list(map(len, reader))
        if len(set(sizes)) != 1:
            raise RuntimeError(
                f'Sizes of datasets {tuple(sizes)} must match.'
            )
        self._inclusions += ['reader']


class Distributed(Nested):
    def __init__(self, reader, world_size, rank):
        assert world_size > 1 and rank >= 0 and world_size > rank, (world_size, rank)
        bufsize = reader._bufsize
        bufsize *= world_size
        # mod = bufsize % world_size
        # if mod > 0:
        #     bufsize = bufsize - mod + world_size
        super().__init__(reader, bufsize, reader._num_threads)
        self.world_size = world_size
        self.rank = rank
        self._counter = rank
        self._inclusions += ['world_size', 'rank', '_counter']

    @overrides
    def _fill_buffer(self):
        super()._fill_buffer()
        buffer = list(self._buffer)
        buffer = buffer[self.rank:None:self.world_size]
        size = len(buffer)
        self._buffer = deque(buffer)
        return size
