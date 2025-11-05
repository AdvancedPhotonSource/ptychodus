from enum import Enum


class StandardFileLayout(str, Enum):
    DIFFRACTION = 'diffraction.h5'
    FLUORESCENCE_IN = 'fluorescence-in.h5'
    FLUORESCENCE_OUT = 'fluorescence-out.h5'
    PRODUCT_IN = 'product-in.h5'
    PRODUCT_OUT = 'product-out.h5'
    SETTINGS = 'settings.ini'
