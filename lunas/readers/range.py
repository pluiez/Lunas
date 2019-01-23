from typing import Any

from overrides import overrides

from lunas.readers.base import Reader


class Range(Reader):
    def __init__(self, start: int, stop: int = None, step: int = None, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(buffer_size, num_threads)
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1

        self.start = start
        self.stop = stop
        self.step = step

        self._range = None

        self._exclusions += ['_range']

    @overrides
    def size(self) -> int:
        import math
        return int(math.ceil((self.stop - self.start) / self.step))

    def _reset(self):
        start, stop, step = self.start, self.stop, self.step
        self._range = iter(range(start, stop, step))

    @overrides
    def reset_cursor(self):
        super().reset_cursor()
        self._reset()

    @overrides
    def next(self) -> Any:
        return next(self._range)
