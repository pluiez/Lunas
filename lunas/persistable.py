import abc
from typing import Dict


class Persistable(object):
    """An interface to introduce persistence method.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def state_dict(self) -> Dict:
        """Returns a dictionary representing the state of the class.

        The returned state dictionary should contain picklable attributes of `self`,
        excluding callable function object.

        Returns:
            A dictionary of `Dict` type represent, which represents the persistent state of `self`.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict) -> None:
        """Restores the state by a given state.

        Sets attributes of `self` by dictionary.

        Args:
            state_dict: A `Dict`.

        """
        raise NotImplementedError
