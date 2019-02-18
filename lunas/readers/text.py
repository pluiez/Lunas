from typing import Any

from overrides import overrides

from lunas.readers.base import BaseReader


class TextLine(BaseReader):
    def __init__(self, filename: str, bufsize: int = 10000, num_threads: int = 1):
        super().__init__(bufsize, num_threads)
        self._filename = filename
        self._fd = None
        self._iterator = None
        self._inclusions += ['_filename', '_num_line']
        self._num_line = -1

    @overrides
    def size(self) -> int:
        if self._num_line < 0:
            n = 0
            with open(self._filename) as r:
                for _ in r:
                    n += 1
            self._num_line = n
        return self._num_line

    @overrides
    def next(self) -> Any:
        line = next(self._iterator)
        return line

    @overrides
    def _reset_cursor(self):
        self._reset()
        super()._reset_cursor()

    def _reset(self):
        self._finalize()
        self._fd = open(self._filename)
        self._iterator = iter(self._fd)

    @overrides
    def _finalize(self):
        if self._fd is not None and not self._fd.closed:
            self._fd.close()
        self._iterator = None
        super()._finalize()
