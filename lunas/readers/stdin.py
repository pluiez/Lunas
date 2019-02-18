import sys

from overrides import overrides

from lunas.readers.base import BaseReader


class Stdin(BaseReader):
    def __init__(self, bufsize: int = 10000, num_threads: int = 1,sentinel=''):
        super().__init__(bufsize, num_threads)
        self._iterator = None
        self._sentinel=sentinel
        self._num_line = 0

    @overrides
    def size(self) -> int:
        return sys.maxsize

    @overrides
    def next(self):
        line=next(self._iterator)
        self._num_line += 1
        return line

    @overrides
    def _reset_cursor(self):
        self._reset()
        super()._reset_cursor()

    def _reset(self):
        self._iterator = iter(sys.stdin.readline, self._sentinel)

    @overrides
    def _finalize(self):
        self._num_line = 0
        self._iterator = None
        super()._finalize()
