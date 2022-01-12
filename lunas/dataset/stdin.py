import os
import sys

import lunas.dataset.core as core

__all__ = ['Stdin']


class Stdin(core.Dataset):
    """Stdin dataset.

    This dataset wraps `sys.stdin`.

    Never use this dataset in multiprocessing context in order not to observe unexpected behaviours since
    the correctness is not guaranteed.
    """

    def __init__(self, sentinel=None, name: str = None):
        super().__init__(name)
        if sentinel is None:
            sentinel = ''
        self._sentinel = sentinel + os.linesep
        self._resumable = False

    def generator(self):
        for x in sys.stdin:
            if self._sentinel == x:
                break
            yield x
