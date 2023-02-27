from abc import ABC, abstractmethod, abstractproperty


class Action(ABC):
    '''interface for workflow actions'''

    @abstractproperty
    def name(self) -> str:
        '''returns a descriptive name for the action'''
        pass

    @abstractmethod
    def __call__(self) -> None:
        '''performs the action'''
        pass
