from typing import List, Dict, Callable, Any

from overrides import overrides

from lunas.batch import Batch
from lunas.dataset.base import BaseDataset
from lunas.persistable import Persistable
from lunas.utils import get_state_dict, load_state_dict


class Iterator(Persistable):
    """An iterator that iterates through a `Reader`.

    This class performs multi-pass iterations over the dataset and maintains
    the iteration state.
    """

    def __init__(self, dataset: BaseDataset, batch_size,
                 sample_size_fn: Callable[[Any], int] = None,
                 collate_fn: Callable[[List[Any]], Any] = None,
                 dist_world_size=1,
                 dist_local_rank=0,
                 drop_tail=False):
        """Initialize the iterator.

        Args:
            dataset: A `Reader` object.
            batch_size: A `int` scalar that limits the size of returned batch.
            sample_size_fn: (Optional.) A callable function that calculates size for each sample.
                The size of each sample will then be summed up as the size of the batch. If not
                specified, default to 1 for each sample, which is equivalent to `lambda sample: 1`.
            collate_fn: (Optional.) A callable function that converts a list of samples to model inputs.
            drop_tail: (Optional.) Whether the last samples of the dataset that cannot fill a batch should be dropped.
        """
        super().__init__()
        # bookkeeping params
        self._step_in_epoch = 0
        self._step = 0
        self._epoch = 0

        self._dataset = dataset

        self._batch_size = batch_size

        self._sample_size_fn = sample_size_fn
        self._collate_fn = collate_fn

        self._dist_world_size = dist_world_size
        self._dist_local_rank = dist_local_rank

        self._drop_tail = drop_tail

        self._inclusions = ['_inclusions', '_step', '_step_in_epoch', '_epoch', '_dataset']

        self.initialize()

    @property
    def step_in_epoch(self):
        return self._step_in_epoch

    @property
    def step(self):
        return self._step

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        return self._batch_size

    def set_batch_size(self, batch_size) -> None:
        """Allows dynamic batch size at runtime.

        Args:
            batch_size: A `int` scalar.

        """
        self._batch_size = batch_size

    def iter_epoch(self, before_epoch=None, after_epoch=None):
        """Iterate through the dataset for one epoch.

        For the last batch, it will be dropped if its size is smaller
        than 2/3 of the specified batch size.

        """
        if before_epoch is not None and self.step_in_epoch == 0:
            before_epoch()

        batch = Batch(self.batch_size, self._sample_size_fn)
        while True:
            try:
                while batch.add(next(self._dataset)):
                    pass
            except StopIteration:
                break
            else:

                if self._step % self._dist_world_size == self._dist_local_rank:
                    yield batch
                else:
                    del batch
                batch = Batch(self.batch_size, self._sample_size_fn)
                self._step += 1
                self._step_in_epoch += 1

        if not self._drop_tail and batch.num_samples > 0:
            yield batch
            self._step += 1
            self._step_in_epoch += 1

        if after_epoch is not None:
            after_epoch()

        self._epoch += 1
        self._step_in_epoch = 0

    def initialize(self):
        self._step_in_epoch = 0
        self._step = 0
        self._epoch = 0
        self._dataset = iter(self._dataset)

    def while_true(self, predicate: Callable[[], bool], before_epoch=None, after_epoch=None):
        """Iterates through the dataset by a given stopping criteria.

        Args:
            predicate: A callable function. This function is evaluated to determine
            whether iteration should continue or not.
            before_epoch:
            after_epoch:

        Returns:
            (batch, inputs): A `Tuple` consists of a `Batch` object and model inputs. When `self.collate_fn`
                is None, the returned `inputs` is also None.
        """
        epoch_iter = self.iter_epoch(before_epoch, after_epoch)

        if predicate is not None:
            while predicate():
                try:
                    batch = next(epoch_iter)
                except StopIteration:
                    self._dataset = iter(self._dataset)
                    epoch_iter = self.iter_epoch(before_epoch, after_epoch)
                    continue

                yield batch
        else:
            for batch in epoch_iter:
                yield batch

    @overrides
    def state_dict(self) -> Dict:
        return get_state_dict(self, recursive=True, inclusions=self._inclusions)

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        load_state_dict(self, state_dict)

    def __call__(self, while_predicate: Callable[[], bool] = None, before_epoch=None, after_epoch=None):
        return self.while_true(while_predicate, before_epoch, after_epoch)
