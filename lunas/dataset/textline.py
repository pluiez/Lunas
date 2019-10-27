import itertools
from pathlib import Path

from .core import Dataset


class TextLine(Dataset):

    def __init__(self, filename: str, name: str = None):
        super().__init__(name)
        filename = Path(filename)
        assert filename.exists()
        self._filename: Path = filename
        self._size = self.count_line(filename)

    @staticmethod
    def count_line(filename):
        f = open(filename, 'rb')
        buf_gen = itertools.takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in itertools.repeat(None)))
        n = 0
        end_is_newline = True
        for buf in buf_gen:
            m = buf.count(b'\n')
            n += m
            if m > 0:
                end_is_newline = buf.rindex(b'\n') == len(buf) - 1
        n += int(not end_is_newline)
        return n

    def __len__(self):
        return self._size

    def generator(self):
        with self._filename.open() as r:
            for l in r:
                yield l
