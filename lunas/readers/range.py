from overrides import overrides

from lunas.readers.base import BaseReader
import itertools
import sys
class Range(BaseReader):
    def __init__(self, start: int=None, stop: int = None, step: int = None, bufsize: int = 10000, num_threads: int = 1):
        super().__init__(bufsize, num_threads)
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1

        self._start = start
        self._stop = stop
        self._step = step

        self._range = None

        self._inclusions += ['_range']

    @overrides
    def size(self) -> int:
        import math
        return int(math.ceil((self._stop - self._start) / self._step))

    @overrides
    def next(self):
        return next(self._range)

    def _reset(self):
        start, stop, step = self._start, self._stop, self._step
        self._range = iter(range(start, stop, step))

    @overrides
    def _reset_cursor(self):
        super()._reset_cursor()
        self._reset()


class Count(BaseReader):
    def __init__(self, start: int=0, step: int = 1, bufsize: int = 10000, num_threads: int = 1):
        super().__init__(bufsize, num_threads)

        self._start = start
        self._step = step

        self._count=None
        self._inclusions += ['_count']

    @overrides
    def size(self) -> int:
        return sys.maxsize

    @overrides
    def next(self):
        return next(self._count)

    def _reset(self):
        self._count= itertools.count(self._start,self._step)

    @overrides
    def _reset_cursor(self):
        super()._reset_cursor()
        self._reset()

