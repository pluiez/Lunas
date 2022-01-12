from __future__ import annotations

import glob
import itertools
import math
import pathlib
from typing import *

import lunas.dataset.core as core

__all__ = ['Array', 'Range', 'Zip', 'Glob']


class Array(core.Dataset, core.Sizable):
    """A dataset that wraps around any data as an iterable array.

    This dataset accepts any data. Note that the iterable data will be converted to an object with give array_type.
    """

    def __init__(self, data: Any, array_type: TypeVar('T') = list, name: str = None):
        super().__init__(name)
        self._data = array_type(data)
        self._length = None

    @property
    def length(self):
        if self._length is None:
            self._length = len(self._data)
        return self._length

    def generator(self):
        for x in self._data:
            yield x


class Range(core.Dataset, core.Sizable):
    """Range dataset.

    This dataset simulates the builtin `range` function.
    """

    def __init__(self, start: int, stop: int = None, step: int = None, name: str = None):
        super().__init__(name)
        if stop is None:
            stop = start
            start = 0

        step = step or 1

        self._start = start
        self._stop = stop
        self._step = step

        self._length = None

    @property
    def length(self):
        if self._length is None:
            start, stop, step = self._start, self._stop, self._step
            self._length = max(0, int(math.ceil((stop - start) / step)))
        return self._length

    def generator(self):
        for x in range(self._start, self._stop, self._step):
            yield x


class Zip(core.NestedN, core.Sizable):
    """Zip multiple dataset.

    This dataset zips multiple datasets, potentially with different sizes.
    """

    def __init__(self, datasets: Iterable[core.Dataset], mode: str = '=', padding: bool = False,
                 name: str = None):
        """Initialises the dataset.

        Args:
            datasets: the datasets to zip.
            mode: a character, available options include '=' '<' and '>'.
                '=' requires the datasets to have the same sizes;
                '<' behaves similarly to the builtin `zip`, which truncate the bigger datasets to align with the
                smallest one;
                '>' is similar to `itertools.zip_longest`, fill the smaller datasets with strategy specified by
                `padding`.
            padding: a boolean value that determines how to pad the small datasets when they are exhausted.
                A `False` will produce `None` as padding, while `True` will continue producing elements from smaller
                datasets. Only works when mode is '>'.
            name: name of the dataset.
        """
        super().__init__(datasets, name)
        if mode not in ('=', '<', '>'):
            raise ValueError(f'Unknown mode: {mode}')

        self._length = None

        if mode == '=':
            sizes = tuple(map(len, datasets))
            if len(set(sizes)) > 1:
                raise RuntimeError(f'Datasets must have exactly the same sizes ({tuple(sizes)}).')

            self._length = sizes[0]

        if not isinstance(padding, bool):
            raise ValueError(f'padding ({padding}) must be a bool value.')

        self._mode = mode
        self._padding = padding

    @property
    def length(self):
        if self._length is None:
            sizes = tuple(map(len, self._datasets))
            if self._mode == '<':
                self._length = min(sizes)
            elif self._mode == '>':
                self._length = max(sizes)
            else:
                raise ValueError(f'Unknown mode: {self._mode}')
        return self._length

    def generator(self):
        if self._mode in ['=', '<']:
            for x in zip(*self._datasets):
                yield x
        else:
            if not self._padding:
                it = itertools.zip_longest(*self._datasets)
                for x in it:
                    yield x
            else:
                # DO NOT USE itertools.cycle EVER!
                # Since itertools.cycle actually stores all elements after one-time iteration,
                # this is super-memory-consuming and breaks the internal state maintenance of a `Dataset`.
                datasets = [itertools.chain.from_iterable(itertools.repeat(d))
                            if len(d) < len(self) else d
                            for d in self._datasets]
                # Additionally, islice is used here. zip will stop when one iterator raises StopIteration,
                # so any datasets before it will unexpectedly advance one element.
                for x in itertools.islice(zip(*datasets), len(self)):
                    yield x


class Glob(core.Dataset, core.Sizable):
    """Glob dataset.

    This dataset uses standard glob module to wrap matched directories/files for given pattern into a dataset.
    """

    def __init__(self, pattern: str, recursive: bool = False, expand_user: bool = True, name: str = None):
        """Initialises the dataset.

        Args:
            pattern: a glob pattern.
            recursive: whether matches recursively.
            expand_user: whether expands the user home path.
            name: name of the dataset.
        """
        super().__init__(name)
        self._pattern = str(pathlib.Path(pattern).expanduser() if expand_user else pathlib.Path(pattern))
        self._files = sorted(glob.glob(self._pattern, recursive=recursive))
        self._length = len(self._files)

    @property
    def length(self):
        return self._length

    def generator(self):
        for x in self._files:
            yield x
