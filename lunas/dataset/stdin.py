import sys

from overrides import overrides

from lunas.dataset.base import BaseDataset


class Stdin(BaseDataset):
    def __init__(self, bufsize: int = 1, sentinel=''):
        super().__init__(bufsize)
        self._iterator = None
        self._sentinel = sentinel
        self._num_line = 0

    @overrides
    def size(self) -> int:
        return 0

    @overrides
    def next(self):
        line = next(self._iterator)
        self._num_line += 1
        return line

    @overrides
    def initialize(self):
        self._num_line = 0
        self._iterator = iter(sys.stdin.readline, self._sentinel)
        super().initialize()

    @overrides
    def finalize(self):
        self._iterator = None

        super().finalize()
