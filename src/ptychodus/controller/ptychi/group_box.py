from ptychodus.api.parameters import BooleanParameter

from ..parameters import CheckableGroupBoxParameterViewController

__all__ = ['PtyChiCheckableGroupBoxViewController']


class PtyChiCheckableGroupBoxViewController(CheckableGroupBoxParameterViewController):
    """A pty-chi group box that collapses to its title row while unchecked.

    The pty-chi panel stacks two dozen optimization and constraint group boxes, most of
    which are off in any given reconstruction. Collapsing the ones that are off keeps the
    panel to the controls that actually take effect.
    """

    def __init__(self, parameter: BooleanParameter, title: str, *, tool_tip: str = '') -> None:
        super().__init__(parameter, title, tool_tip=tool_tip, collapse_when_unchecked=True)
