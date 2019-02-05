import sys
from typing import Any

from lunas.readers.base import Reader
from overrides import overrides


class Stdin(Reader):
    def __init__(self, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(buffer_size, num_threads)
        self._exclusions += ['_fd']
        self._num_line = 0

    @overrides
    def size(self) -> int:
        return sys.maxsize

    @overrides
    def finalize(self):
        self._num_line = 0
        super().finalize()

    @overrides
    def next(self) -> Any:
        line = sys.stdin.readline()
        if line == '':
            raise StopIteration
        self._num_line += 1
        return line
