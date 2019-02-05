from typing import List, Dict, Callable, Any

from lunas.batch import Batch, Cache
from lunas.persistable import Persistable
from lunas.readers import Reader
from lunas.utils import get_state_dict, load_state_dict
from overrides import overrides


class Iterator(Persistable):
    """An iterator that iterates through a `Reader`.

    This class performs multi-pass iterations over the dataset and maintains
    the iteration state.
    """

    def __init__(self, reader: Reader, batch_size, cache_size: int = 1000, sample_size_fn: Callable[[Any], int] = None,
                 collate_fn: Callable[[List[Any]], Any] = None, sort_cache_by: Callable[[Any], int] = None,
                 sort_batch_by: Callable[[Any], int] = None,
                 drop_tails=False):
        """Initialize the iterator.

        Args:
            reader: A `Reader` object.
            batch_size: A `int` scalar that limits the size of returned batch.
            cache_size: A `int` scalar. Prefetch `cache_size` samples from the `reader` in `self.cache`.
            sample_size_fn: (Optional.) A callable function that calculates size for each sample.
                The size of each sample will then be summed up as the size of the batch. If not
                specified, default to 1 for each sample, which is equivalent to `lambda sample: 1`.
            collate_fn: (Optional.) A callable function that converts a list of samples to model inputs.
            sort_cache_by: (Optional.) A callable function that returns a sorting key for each sample. If not
                specified, leave the cache as it is. The samples will be sorted in ascending order.
            sort_batch_by: (Optional.) A callable function that returns a sorting key for each sample. If not
                specified, leave the batch as it is. The samples will be sorted in ascending order.
            drop_tails: (Optional.) Whether the last samples of the dataset that cannot fill a batch should be dropped.
        """
        super().__init__()
        self._reader = reader
        self._batch_size = batch_size
        self._sample_size_fn = sample_size_fn
        self._collate_fn = collate_fn
        self._cache_size = cache_size
        self._sort_cache_by = sort_cache_by
        self._sort_batch_by = sort_batch_by
        self._drop_tails = drop_tails

        self._sample_size_fn = sample_size_fn
        self._collate_fn = collate_fn

        # bookkeeping params
        self._step_in_epoch = 0
        self._step = 0
        self._epoch = 0

        self._cache = Cache(cache_size, sample_size_fn)
        self._remains = []

        self._exclusions = ['_sample_size_fn', '_collate_fn', '_sort_cache_by', '_sort_batch_by']

        self.check_batch_size(batch_size, cache_size)
        self.reset()

    @property
    def cache_size(self):
        return self._cache_size

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
        self.check_batch_size(batch_size)
        self._batch_size = batch_size

    def check_batch_size(self, batch_size, cache_size=None) -> None:
        """Checks whether batch_size is < cache_size.

        To ensure rationality, batch_size must be < cache_size.

        Args:
            batch_size: A `int` scalar.
            cache_size: A `int` scalar.

        """
        cache_size = cache_size or self._cache_size
        if batch_size > cache_size:
            raise RuntimeError(
                f'Batch size ({batch_size}) should be less than cache size ({cache_size}). '
                f'Please lower the batch size or increase the cache size.'
            )

    def reset(self):
        self._step_in_epoch = 0
        self._step = 0
        self._epoch = 0
        self._remains.clear()
        self._cache.pop_all()  # discard
        self._reader = iter(self._reader)

    def reset_epoch(self):
        self._step_in_epoch = 0
        self._remains.clear()
        self._cache.pop_all()

    def _prepare_batch(self, batch: Batch):
        if self._collate_fn:
            batch.process(self._collate_fn)
        return batch

    def iter_epoch(self, before_epoch=None, after_epoch=None):
        """Iterate through the dataset for one epoch.

        For the last batch, it will be dropped if its size is smaller
        than 2/3 of the specified batch size.

        """
        # self.reset_epoch()
        cache = self._cache
        remains = self._remains

        end_of_epoch = False

        sort_batch = False
        if before_epoch is not None and self._step_in_epoch == 0:
            before_epoch()

        while True:
            batch = Batch(self._batch_size, self._sample_size_fn)
            if cache.effective_size() < self._batch_size * 2 / 3.0:
                if end_of_epoch:
                    # Raise error when the whole dataset cannot form a batch
                    if self._step == 0:
                        raise RuntimeError(
                            f'Size of the dataset ({len(remains)}) '
                            f'is smaller than batch size ({self._batch_size}). '
                            f'Please lower the batch size or '
                            f'check whether the dataset is too small.'
                        )
                    self._reader = iter(self._reader)

                    if self._drop_tails or len(remains) == 0:
                        break
                    else:
                        # The last batch
                        batch.from_list(remains, self._batch_size)
                        batch.from_iter(cache, self._batch_size)
                        batch.sort(self._sort_batch_by or self._sort_cache_by)
                        self._step_in_epoch += 1
                        self._step += 1
                        yield self._prepare_batch(batch)
                        break

                # Consume samples from cache before filling-in
                remains += cache.pop_all()
                try:
                    # Fill cache
                    cache.from_iter(self._reader, raise_when_stopped=True)
                except StopIteration:
                    # Mark as end
                    end_of_epoch = True
                cache.sort(self._sort_cache_by)
            if self._batch_size == self._cache_size:
                # Simply return the cache as a batch to avoid sorting again.
                batch = cache
                cache = Cache(self._cache_size, self._sample_size_fn)
                self._cache = cache
            else:
                if remains:
                    batch.from_list(remains, self._batch_size)
                    sort_batch = True
                size = batch.effective_size()
                batch.from_iter(cache, self._batch_size)
                size_diff = batch.effective_size() - size
                sort_batch = size_diff > 0 or sort_batch

            if batch.effective_size() < self._batch_size:
                remains += batch.pop_all()
            else:
                if sort_batch:
                    batch.sort(self._sort_batch_by or self._sort_cache_by)
                    sort_batch = False

                self._step_in_epoch += 1
                self._step += 1
                yield self._prepare_batch(batch)

        if after_epoch is not None:
            after_epoch()

        self._epoch += 1
        self._step_in_epoch = 0
        raise StopIteration

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
                    epoch_iter = self.iter_epoch(before_epoch, after_epoch)
                    continue

                yield batch
        else:
            for batch in epoch_iter:
                yield batch

    def __call__(self, while_predicate: Callable[[], bool] = None, before_epoch=None, after_epoch=None):
        return self.while_true(while_predicate, before_epoch, after_epoch)

    @overrides
    def state_dict(self) -> Dict:
        return get_state_dict(self, exclusions=self._exclusions, recursive=True)

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        load_state_dict(self, state_dict)
