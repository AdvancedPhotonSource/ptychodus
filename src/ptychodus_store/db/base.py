from __future__ import annotations

from enum import StrEnum

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    'ix': 'ix_%(column_0_label)s',
    'uq': 'uq_%(table_name)s_%(column_0_name)s',
    'ck': 'ck_%(table_name)s_%(constraint_name)s',
    'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
    'pk': 'pk_%(table_name)s',
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class IngestState(StrEnum):
    DISCOVERED = 'DISCOVERED'
    VALID = 'VALID'
    INVALID = 'INVALID'
    MISSING_FILES = 'MISSING_FILES'
    ORPHANED = 'ORPHANED'
