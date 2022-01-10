import random
from typing import *

import lunas.dataset.core as core

__all__ = ['Sampling']


def normalise_weights(weights):
    sum_weight = sum(weights)
    return [w / sum_weight for w in weights]


class Sampling(core.NestedN):
    """Sampling dataset

    Sample examples from multiple datasets by given weights.
    """

    def __init__(self,
                 datasets: Iterable[core.Dataset],
                 weights: Optional[Iterable[float]] = None,
                 virtual_size: Optional[int] = None,
                 name: str = None):
        super().__init__(datasets, name)

        datasets = self._datasets

        if weights and sum(weights) != 1:
            raise ValueError(f'Expected the sum of weights to be 1.0, got {sum(weights)} instead.')
        if virtual_size is not None and virtual_size < 0:
            raise ValueError(f'virtual size should be None or a non-negative value, got {virtual_size} instead.')
        if weights:
            for ds, weight in zip(datasets, weights):
                if len(ds) == 0 and weight > 0:
                    raise ValueError(f'Attempt to sample from an empty dataset '
                                     f'with a non-zero weight ({weight}).')

        sizes = [len(ds) for ds in datasets]

        weights = weights if weights else [1.0 / len(datasets)] * len(datasets)

        max_weight_i, _ = max(enumerate(weights), key=lambda i_weight: i_weight[1])
        if virtual_size is None:
            virtual_size = int(sizes[max_weight_i] / weights[max_weight_i])
            virtual_size = sum(int(virtual_size * weight) for weight in weights)

        self._weights = weights
        self._virtual_size = virtual_size

    def __len__(self):
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
