from __future__ import annotations

import glob
import itertools
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Iterable, Union

import numpy

from .generic import Dataset, Nested


class Array(Dataset):

    def __init__(self, data: Union[numpy.ndarray, Iterable, Iterator], name: str = None):
        super().__init__(name)

        if not isinstance(data, numpy.ndarray) and isinstance(data, Iterator):
            data = list(data)
        self._data = data

    def __len__(self):
        return len(self._data)

    def generator(self):
        if isinstance(self._data, numpy.ndarray):
            it = numpy.nditer(self._data)
        else:
            it = iter(self._data)
        for x in it:
            yield x


class Range(Dataset):
    def __init__(self, start: int, stop: int = None, step: int = None, name: str = None):
        super().__init__(name)
        if stop is None:
            stop = start
            start = 0

        step = step or 1

        self._start = start
        self._stop = stop
        self._step = step

    def __len__(self):
        start, stop, step = self._start, self._stop, self._step
        return abs(int(math.ceil((stop - start) / step)))

    def generator(self):
        for x in range(self._start, self._stop, self._step):
            yield x


class Enumerate(Nested):

    def __init__(self, dataset: Dataset, start: int = 0, name: str = None):
        super().__init__(name)
        self._start = start

    def __len__(self):
        return len(self._dataset)

    def generator(self):
        for x in enumerate(self._dataset, self._start):
            yield x


class Zip(Dataset):

    def __init__(self, datasets: Iterable[Dataset], mode: str = '=', padding: bool = False, name: str = None):
        """
        Zip multiple datasets, potentially with different sizes.
        :param datasets:
        :param name:
        :param mode: a character, available options include '=' '<' and '>'. '=' requires the datasets to have the
            same sizes; '<' behaves similarly to the builtin zip, which strip according to the shortest dataset;
            '>' is similar to itertools.zip_longest.
        :param padding: a boolean value that determines how to pad the shorter datasets when they are exhausted.
            A False will use None as padding, just like itertools.zip_longest. A True value will reiterate over
            the shorter dataset to produce element instead of None padding. Only works when mode is '>'.
        """
        super().__init__(name)
        assert isinstance(datasets, (tuple, list))
        datasets = tuple(datasets)
        assert all(map(lambda x: isinstance(x, Dataset), datasets)), 'datasets must subclass Dataset.'

        sizes = tuple(map(len, datasets))
        if mode == '=':
            if len(set(sizes)) > 1:
                raise RuntimeError(f'Datasets must have exactly the same sizes. Got sizes: {tuple(sizes)}')
            size = sizes[0]
        elif mode == '<':
            size = min(sizes)
        elif mode == '>':
            size = max(sizes)
        else:
            raise NotImplementedError(f'Unknown mode:{mode}')

        self._size = size

        self._datasets = datasets
        self._mode = mode
        self._padding = padding

    def __len__(self):
        return self._size

    def generator(self):
        if self._mode in ['=', '<']:
            for x in zip(*self._datasets):
                yield x
        else:
            it = itertools.zip_longest(*self._datasets)
            if self._padding:
                datasets = [dataset if len(dataset) == len(self) else dataset.repeat() for dataset in self._datasets]
                it = zip(*datasets)
            for x in it:
                yield x


class Glob(Dataset):

    def __init__(self, pattern: str, recursive: bool = False, expand_user: bool = False, name: str = None):
        super().__init__(name)
        pattern = str(Path(pattern).expanduser() if expand_user else Path(pattern))
        self.files = sorted(glob.glob(pattern, recursive=recursive))

    def __len__(self):
        return len(self.files)

    def generator(self):
        for x in self.files:
            yield x
