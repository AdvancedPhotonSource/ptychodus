from enum import auto, Enum


class ScanIndexFilter(Enum):
    '''filters scan points by index'''
    ALL = auto()
    ODD = auto()
    EVEN = auto()

    @property
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        return self.name

    @property
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        return self.name.title()

    def __call__(self, index: int) -> bool:
        '''include scan point if true, exclude otherwise'''
        if self is ScanIndexFilter.ODD:
            return (index % 2 != 0)
        elif self is ScanIndexFilter.EVEN:
            return (index % 2 == 0)

        return True
