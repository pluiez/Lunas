from typing import Any

from overrides import overrides

from lunas.dataset.base import BaseDataset


class TextLine(BaseDataset):
    def __init__(self, filename: str, bufsize: int = 1):
        super().__init__(bufsize)
        self._filename = filename
        self._stream = None
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
        line = next(self._stream)
        return line

    @overrides
    def initialize(self):
        if self._initialized:
            raise IOError('File not closed!')
        self._stream = open(self._filename)
        super().initialize()

    @overrides
    def finalize(self):
        if self._stream is not None and not self._stream.closed:
            self._stream.close()
        self._stream = None
        super().finalize()
