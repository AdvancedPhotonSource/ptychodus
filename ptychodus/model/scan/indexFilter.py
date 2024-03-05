from enum import auto, Enum


class ScanIndexFilter(Enum):
    '''filters scan points by index'''
    ALL = auto()
    ODD = auto()
    EVEN = auto()

    def __call__(self, index: int) -> bool:
        '''include scan point if true, exclude otherwise'''
        if self is ScanIndexFilter.ODD:
            return (index & 1 != 0)
        elif self is ScanIndexFilter.EVEN:
            return (index & 1 == 0)

        return True
