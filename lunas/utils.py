import itertools
from multiprocessing.pool import ThreadPool
from typing import Callable, Any, List


def parallel_map(fn: Callable[[Any], Any], inputs: List[Any], num_thread: int):
    """Applies a function to a list of inputs in parallel.

    Uses a Threading pool to process inputs in parallel.

    Args:
        fn: A `Callable` function that takes an element from `inputs` as argument.
        inputs: A `Iterable` object.
        num_thread: An `int` scalar represents the number of threads.

    Returns:
        A corresponding list of processed inputs.
    """
    if num_thread <= 1:
        return list(map(fn, inputs))
    pool = ThreadPool(num_thread)
    results = pool.map(fn, inputs)

    pool.close()
    pool.join()
    return results


def try_get_attr(instance, attr_name):
    if hasattr(instance, attr_name):
        return getattr(instance, attr_name)
    else:
        return None


def get_state_dict(obj, recursive=True, exclusions=None, inclusions=None):
    keys = vars(obj).keys()
    state_dict = {}
    if exclusions and inclusions:
        raise TypeError('Argument `exclusions` and `inclusions` '
                        'should not be both non-empty at the same time.')

    if exclusions:
        keys = [k for k in keys if k not in exclusions]
    elif inclusions:
        keys = [k for k in keys if k in inclusions]

    props = map(lambda k: getattr(obj, k), keys)

    for key, prop in itertools.zip_longest(keys, props):
        if recursive:
            to_state_dict = try_get_attr(prop, 'state_dict')
            if callable(to_state_dict):
                prop = to_state_dict()
        state_dict[key] = prop

    return state_dict


def load_state_dict(obj, state_dict):
    for key, val in state_dict.items():
        prop = getattr(obj, key)
        load_state_dict_ = try_get_attr(prop, 'load_state_dict')
        if callable(load_state_dict_):
            load_state_dict_(val)
        else:
            setattr(obj, key, val)
