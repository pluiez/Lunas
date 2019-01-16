from typing import List, Dict, Callable, Any

from overrides import overrides

from lunas.batch import Batch, Cache
from lunas.interface import Persistable
from lunas.reader import Reader
from lunas.utils import get_state_dict, load_state_dict


class DataIterator(Persistable):
    """An iterator that iterates through a `Reader`.

    This class performs multi-pass iterations over the dataset and maintains
    the iteration state.
    """

    def __init__(self, reader: Reader, batch_size, cache_size: int = 1000, sample_size_fn: Callable[[Any], int] = None,
                 collate_fn: Callable[[List[Any]], Any] = None, sort_desc_by: Callable[[Any], int] = None):
        """Initialize the iterator.

        Args:
            reader: A `Reader` object.
            batch_size: A `int` scalar that limits the size of returned batch.
            cache_size: A `int` scalar. Prefetch `cache_size` samples from the `reader` in `self.cache`.
            sample_size_fn: (Optional.) A callable function that calculates size for each sample.
                The size of each sample will then be summed up as the size of the batch. If not
                specified, default to 1 for each sample, which is equivalent to `lambda sample: 1`.
            collate_fn: (Optional.) A callable function that converts a list of samples to model inputs.
            sort_desc_by: (Optional.) A callable function that returns a sorting key for each sample. If not
                specified, leave the batch as it is.
        """
        super().__init__()
        self.reader = reader
        self.batch_size = batch_size
        self.sample_size_fn = sample_size_fn
        self.collate_fn = collate_fn
        self.cache_size = cache_size
        self.sort_desc_by = sort_desc_by

        if collate_fn is None:
            collate_fn = lambda samples: None

        self.sample_size_fn = sample_size_fn
        self.collate_fn = collate_fn

        # bookkeeping params
        self.step_in_epoch = -1
        self.step = -1
        self.epoch = 0

        self.cache = Cache(cache_size, sample_size_fn)
        self.remains = []

        self._exclusions = ['sample_size_fn', 'collate_fn', 'sort_desc_by']

        self.check_batch_size(batch_size, cache_size)
        self.reset()

    def set_batch_size(self, batch_size) -> None:
        """Allows dynamic batch size at runtime.

        Args:
            batch_size: A `int` scalar.

        """
        self.check_batch_size(batch_size)
        self.batch_size = batch_size

    def check_batch_size(self, batch_size, cache_size=None) -> None:
        """Checks whether batch_size is < cache_size.

        To ensure rationality, batch_size must be < cache_size.

        Args:
            batch_size: A `int` scalar.
            cache_size: A `int` scalar.

        """
        cache_size = cache_size or self.cache_size
        if batch_size > cache_size:
            raise RuntimeError(f'Batch size should be less than cache size. '
                               f'Got batch_size = {batch_size} and cache_size = {cache_size}.')

    def reset(self):
        self.step_in_epoch = -1
        self.step = -1
        self.epoch = 0
        self.remains.clear()
        self.cache.pop_all()
        self.reader = iter(self.reader)

    def reset_epoch(self):
        self.step_in_epoch = -1
        self.remains.clear()
        self.cache.pop_all()

    def iter_epoch(self):
        """Iterate through the dataset for one epoch.

        For the last batch, it will be dropped if its size is smaller
        than 2/3 of the specified batch size.

        """
        # self.reset_epoch()
        cache = self.cache
        remains = self.remains

        end_of_epoch = False
        sort_batch = False
        while True:
            batch = Batch(self.batch_size, self.sample_size_fn)
            if cache.effective_size() < self.batch_size * 2 / 3.0:
                if end_of_epoch:
                    self.reader = iter(self.reader)
                    break
                # Consume samples from cache before filling-in
                remains += cache.pop_all()
                try:
                    # Fill cache
                    cache.from_iter(self.reader, raise_when_stopped=True)
                except StopIteration:
                    # Mark as end
                    end_of_epoch = True
                cache.sort(self.sort_desc_by)

            if self.batch_size == self.cache_size:
                # Simply return the cache such that the samples
                # can be desorted later.
                batch = cache
                cache = Cache(self.cache_size, self.sample_size_fn)
                self.cache = cache
                self.step_in_epoch += 1
                self.step += 1
                yield (batch, self.collate_fn(batch.samples))

            else:
                if remains:
                    batch.from_list(remains, self.batch_size)
                    sort_batch = True
                batch.from_iter(cache, self.batch_size)

                if batch.effective_size() < self.batch_size:
                    remains += batch.pop_all()
                else:
                    if sort_batch:
                        batch.sort(self.sort_desc_by)
                        sort_batch = False

                    self.step_in_epoch += 1
                    self.step += 1
                    yield (batch, self.collate_fn(batch.samples))

        self.epoch += 1
        self.step_in_epoch = -1
        raise StopIteration

    def while_true(self, predicate: Callable[[], bool]):
        """Iterates through the dataset by a given stopping criteria.

        Args:
            predicate: A callable function. This function is evaluated to determine
            whether iteration should continue or not.

        Returns:
            (batch, inputs): A `Tuple` consists of a `Batch` object and model inputs. When `self.collate_fn`
                is None, the returned `inputs` is also None.
        """
        epoch_iter = self.iter_epoch()

        while predicate():
            try:
                batch = next(epoch_iter)
            except StopIteration:
                # self.epoch += 1
                # self.step_in_epoch = -1
                epoch_iter = self.iter_epoch()
                continue

            yield batch

    def __call__(self, while_predicate: Callable[[], bool]):
        return self.while_true(while_predicate)

    @overrides
    def state_dict(self) -> Dict:
        return get_state_dict(self, exclusions=self._exclusions, recursive=True)

    @overrides
    def load_state_dict(self, state_dict: Dict) -> None:
        load_state_dict(self, state_dict)
