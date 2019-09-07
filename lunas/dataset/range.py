import itertools
import math

from overrides import overrides

from lunas.dataset.base import BaseDataset


class Range(BaseDataset):
    def __init__(self, start: int = None, stop: int = None, step: int = None, bufsize: int = 1):
        super().__init__(bufsize)
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1

        self._start = start
        self._stop = stop
        self._step = step

        self._range = None

    @property
    def size(self) -> int:
        return int(math.ceil((self._stop - self._start) / self._step))

    @overrides
    def next(self):
        if not self._initialized:
            raise StopIteration
        return next(self._range)

    @overrides
    def initialize(self):
        self._range = iter(range(self._start, self._stop, self._step))
        super().initialize()

    @overrides
    def finalize(self):
        self._range = None
        super().finalize()


class Count(BaseDataset):
    def __init__(self, start: int = 0, step: int = 1, bufsize: int = 1):
        super().__init__(bufsize)

        self._start = start
        self._step = step

        self._count = None

    @overrides
    def size(self) -> int:
        return 0

    @overrides
    def next(self):
        return next(self._count)

    @overrides
    def initialize(self):
        self._count = itertools.count(self._start, self._step)
        super().initialize()

    @overrides
    def finalize(self):
        self._count = None

        super().finalize()
