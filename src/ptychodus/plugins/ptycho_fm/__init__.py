"""PtychoFM, relocated ahead of the unit that makes it registrable.

No ``register_plugins`` yet: constructing the library still needs a settings
registry, and a plugin is registered as an instance. The hook arrives once the
backend is configured from the options it is handed per call.
"""

from .core import PtychoFMReconstructorLibrary

__all__ = [
    'PtychoFMReconstructorLibrary',
]
