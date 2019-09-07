from typing import List, Dict, Callable, Any

import numpy
from overrides import overrides

from lunas.dataset.base import BaseDataset
from lunas.dataset.range import Count


class Nested(BaseDataset):
    def __init__(self, *datasets: BaseDataset, bufsize: int = None):
        bufsize = bufsize or max(d._bufsize for d in datasets)
        super().__init__(bufsize)
        self.datasets: List[BaseDataset] = list(datasets)
        if len(datasets) == 1:
            self._buffer_callbacks.append(self._flatten)
        self._next_fns = [{'next': d.__next__, 'buffered_next': d._buffered_next} for d in datasets]
        self._inclusions += ['datasets']

    def _flatten(self, buffer):
        buffer = [x for (x,) in buffer]
        return buffer

    @property
    def size(self) -> int:
        return len(self.datasets[0])

    @overrides
    def next(self):
        # sample = [d.next() for d in self.datasets]
        sample = [d._buffered_next() for d in self.datasets]
        return sample

    @overrides
    def enable_processing(self, enable: bool):
        super().enable_processing(enable)
        key = 'next' if enable else 'buffered_next'
        for i, d in enumerate(self.datasets):
            setattr(d, '_buffered_next', self._next_fns[i][key])

    @overrides
    def state_dict(self) -> Dict:
        state = super().state_dict()
        state['datasets'] = [d.state_dict() for d in self.datasets]
        return state

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        for d, state in zip(self.datasets, state_dict['datasets']):
            d.load_state_dict(state)
        del state_dict['datasets']

        super().load_state_dict(state_dict)

    # @overrides
    # def process(self, buffer=None):
    #     buffer = self._buffer if buffer is None else buffer
    #     zip_ = itertools.zip_longest
    # processed individual buffers
    # buffers = [d.process(buf) for d, buf in zip_(self.datasets, zip_(*buffer))]
    #
    # if len(self.datasets) == 1:
    #     return super().process(buffers[0])
    # else:
    #     return super().process(zip_(*buffers))
    # if len(self.datasets) == 1:
    #     return super().process([x[0] for x in buffer])
    # else:
    #     return super().process(buffer)

    @overrides
    def initialize(self):
        for d in self.datasets:
            d.initialize()
        super().initialize()

    @overrides
    def finalize(self):
        for d in self.datasets:
            d.finalize()
        super().finalize()


class Zip(Nested):
    def __init__(self, *datasets: BaseDataset, bufsize: int = None, strict=True):
        super().__init__(*datasets, bufsize=bufsize)
        if strict:
            sizes = list(map(len, datasets))
            if min(sizes) != max(sizes):
                raise RuntimeError(
                    f'Sizes of datasets ({tuple(sizes)}) must match.'
                )


class Shuffle(Nested):
    def __init__(self, dataset: BaseDataset, bufsize: int = None):
        super().__init__(dataset, bufsize=bufsize)
        self._buffer_callbacks.append(self._shuffle)

    def _shuffle(self, buffer):
        self._random_state = numpy.random.get_state()
        numpy.random.shuffle(buffer)

        return buffer


class Sort(Nested):
    def __init__(self, dataset: BaseDataset, bufsize: int = None, sort_key_fn: Callable[[Any], Any] = None):
        super().__init__(dataset, bufsize=bufsize)
        self._sort_key_fn = sort_key_fn
        self._buffer_callbacks.append(self._sort)

    def _sort(self, buffer):
        buffer = sorted(buffer, key=self._sort_key_fn)
        return buffer


class InvertibleSort(Nested):
    def __init__(self, dataset: BaseDataset, bufsize: int = None, sort_key_fn: Callable[[Any], Any] = None):
        dataset = Enumerate(dataset)
        super().__init__(dataset, bufsize=bufsize)
        self._sort_key_fn = sort_key_fn
        self._buffer_callbacks.append(self._sort)

    def _sort(self, buffer):
        idx2sample = dict(buffer)
        sort_idx = sorted(idx2sample.keys(), key=lambda i: self._sort_key_fn(idx2sample[i]))
        return [(i, idx2sample[i]) for i in sort_idx]


class Enumerate(Zip):
    def __init__(self, dataset: BaseDataset, start=0, step=1, bufsize: int = None):
        super().__init__(Count(start, step, bufsize or dataset._bufsize), dataset, bufsize=bufsize, strict=False)
