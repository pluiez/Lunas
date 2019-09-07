import itertools


def try_get_attr(instance, attr_name, value=None):
    if hasattr(instance, attr_name):
        return getattr(instance, attr_name)
    else:
        return value


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
