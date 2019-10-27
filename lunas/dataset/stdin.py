import sys

from .core import Dataset


class Stdin(Dataset):

    def __init__(self, sentinel: str = '', name: str = None):
        super().__init__(name)
        self._sentinel = sentinel

    def __len__(self):
        return sys.maxsize

    def generator(self):
        for x in sys.stdin:
            if not x.strip():
                break
            yield x
