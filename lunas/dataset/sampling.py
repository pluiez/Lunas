import random
from typing import *

import lunas.dataset.core as core

__all__ = ['Sampling']


class Sampling(core.NestedN, core.Sizable):
    """Sampling dataset

    This dataset sample examples from multiple datasets by given weights.
    """

    def __init__(self, datasets: Iterable[core.Dataset], weights: Iterable[float] = None, sizes: Iterable[int] = None, virtual_size: int = None,
                 name: str = None):
        super().__init__(datasets, name)

        datasets = self._datasets

        if weights:
            if sum(weights) != 1:
                raise ValueError(f'The sum of weights ({sum(weights)}) must be 1.0.')
            if min(weights) < 0:
                raise ValueError(f'weights ({min(weights)}) must be non-negative value.')

        if virtual_size is not None and virtual_size < 0:
            raise ValueError(f'virtual_size ({virtual_size}) must be a non-negative value or None.')

        weights = weights if weights else [1.0 / len(datasets)] * len(datasets)

        max_weight_i, _ = max(enumerate(weights), key=lambda i_weight: i_weight[1])
        if virtual_size is None:
            if sizes is None:
                sizes = [len(ds) for ds in datasets]
            virtual_size = int(sizes[max_weight_i] / weights[max_weight_i])
            virtual_size = sum(int(virtual_size * weight) for weight in weights)

        self._weights = weights
        self._virtual_size = virtual_size

    @property
    def length(self):
        return self._virtual_size

    def generator(self):
        indices = [i for i in range(len(self._datasets))]
        weights = [w for w in self._weights]
        datasets = [iter(dataset) for dataset in self._datasets]
        for _ in range(self._virtual_size):
            if len(indices) > 1:
                i = random.choices(indices, weights)[0]
            else:
                i = indices[0]
            try:
                yield next(datasets[i])
            except StopIteration:
                datasets[i] = iter(self._datasets[i])
                yield next(datasets[i])
