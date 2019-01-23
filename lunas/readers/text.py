from typing import TextIO, Any

from overrides import overrides

from lunas.readers.base import Reader


class TextLine(Reader):
    def __init__(self, filename: str, buffer_size: int = 10000, num_threads: int = 1):
        super().__init__(buffer_size, num_threads)
        self._filename = filename
        self._fd: TextIO = None
        self._exclusions += ['_fd']
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
    def reset_cursor(self):
        self._reset()
        super().reset_cursor()

    def _reset(self):
        self.finalize()
        self._fd = open(self._filename)

    @overrides
    def finalize(self):
        if self._fd is not None and not self._fd.closed:
            self._fd.close()
        super().finalize()

    @overrides
    def next(self) -> Any:
        line = self._fd.readline()
        if line == '':
            raise StopIteration
        return line
