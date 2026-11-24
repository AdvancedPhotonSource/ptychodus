"""Markdown and LaTeX math to sanitized HTML, rendered on the server.

The web UI inserts what this returns as HTML, so everything passes through one
allowlist sanitizer last: basic HTML and MathML presentation elements, no scriptable
attribute, no images (a model-written image URL would fetch on display) and links only
to http, https, mailto and relative URLs. latex2mathml escapes nothing, so its output is
as untrusted as the text it came from.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Sequence

import latex2mathml.converter
import nh3
from markdown_it import MarkdownIt
from markdown_it.renderer import RendererProtocol
from markdown_it.token import Token
from markdown_it.utils import EnvType, OptionsDict
from mdit_py_plugins.dollarmath import dollarmath_plugin

logger = logging.getLogger(__name__)

__all__ = [
    'BlockRenderer',
    'MessageRenderer',
    'utf16_length',
]


def utf16_length(text: str) -> int:
    """The length of `text` in UTF-16 code units, as a JavaScript string measures it."""
    return len(text.encode('utf-16-le')) // 2


def _render_math(content: str, config: dict[str, object]) -> str:
    display = 'block' if config.get('display_mode') else 'inline'

    try:
        return latex2mathml.converter.convert(content, display=display)
    except Exception:  # latex2mathml raises a variety of exceptions on malformed TeX
        logger.debug('Showing math latex2mathml cannot convert as source: %r', content)
        return f'<code class="math-error">{html.escape(content)}</code>'


def _render_math_inline_double(
    renderer: RendererProtocol,
    tokens: Sequence[Token],
    idx: int,
    options: OptionsDict,
    env: EnvType,
) -> str:
    # dollarmath wraps $$...$$ written inside a paragraph in a <div>, which a <p> cannot
    # hold, so the browser would close the paragraph around it.
    content = _render_math(str(tokens[idx].content).strip(), {'display_mode': True})
    return f'<span class="math display">{content}</span>'


class MessageRenderer:
    """Renders message text as sanitized HTML."""

    _HTML_TAGS = frozenset(
        {
            'a',
            'blockquote',
            'br',
            'code',
            'div',
            'em',
            'h1',
            'h2',
            'h3',
            'h4',
            'h5',
            'h6',
            'hr',
            'li',
            'ol',
            'p',
            'pre',
            's',
            'span',
            'strong',
            'table',
            'tbody',
            'td',
            'th',
            'thead',
            'tr',
            'ul',
        }
    )
    _MATHML_TAGS = frozenset(
        {
            'math',
            'menclose',
            'mfrac',
            'mi',
            'mmultiscripts',
            'mn',
            'mo',
            'mover',
            'mpadded',
            'mphantom',
            'mprescripts',
            'mroot',
            'mrow',
            'ms',
            'mspace',
            'msqrt',
            'mstyle',
            'msub',
            'msubsup',
            'msup',
            'mtable',
            'mtd',
            'mtext',
            'mtr',
            'munder',
            'munderover',
            'none',
        }
    )
    # Presentation attributes only: MathML's href and the xlink attributes are left out,
    # because each is a link a script URL could ride.
    _MATHML_ATTRIBUTES = frozenset(
        {
            'accent',
            'accentunder',
            'columnalign',
            'columnlines',
            'columnspacing',
            'columnspan',
            'depth',
            'display',
            'displaystyle',
            'fence',
            'form',
            'frame',
            'height',
            'largeop',
            'linebreak',
            'linethickness',
            'lspace',
            'mathcolor',
            'mathvariant',
            'maxsize',
            'minsize',
            'movablelimits',
            'notation',
            'rowalign',
            'rowlines',
            'rowspacing',
            'rowspan',
            'rspace',
            'scriptlevel',
            'separator',
            'stretchy',
            'symmetric',
            'voffset',
            'width',
        }
    )

    def __init__(self) -> None:
        self._markdown = MarkdownIt('commonmark', {'html': False}).enable(
            ['table', 'strikethrough']
        )
        # No space inside the delimiters and no digit after the closing one, so prices
        # like "$5 and $10" stay text.
        dollarmath_plugin(
            self._markdown,
            allow_labels=False,
            allow_space=False,
            allow_digits=False,
            double_inline=True,
            renderer=_render_math,
        )
        self._markdown.add_render_rule('math_inline_double', _render_math_inline_double)
        self._attributes = {
            'a': {'href', 'title'},
            'ol': {'start'},
            **{tag: set(self._MATHML_ATTRIBUTES) for tag in self._MATHML_TAGS},
        }

    def render(self, text: str) -> str:
        """Return `text`, read as Markdown with $-delimited math, as sanitized HTML."""
        return self.sanitize(self._markdown.render(text))

    def sanitize(self, unsafe_html: str) -> str:
        """Return `unsafe_html` reduced to the allowlist."""
        return nh3.clean(
            unsafe_html,
            tags=set(self._HTML_TAGS | self._MATHML_TAGS),
            attributes=self._attributes,
            allowed_classes={
                'code': {'math-error'},
                'div': {'math', 'block'},
                'span': {'math', 'inline', 'display'},
            },
            url_schemes={'http', 'https', 'mailto'},
            link_rel='noopener noreferrer',
            strip_comments=True,
        )

    def count_complete_lines(self, text: str) -> int:
        """The number of leading lines of `text` that form complete top-level blocks.

        Every top-level block but the last is complete, since a later block has begun.
        The last may still grow: an unclosed fence or ``$$`` runs to the end of the
        text, and a paragraph or list continues until a blank line and another block.
        """
        starts = [
            token.map[0]
            for token in self._markdown.parse(text)
            if token.level == 0 and token.nesting >= 0 and token.map is not None
        ]
        return starts[-1] if len(starts) > 1 else 0


class BlockRenderer:
    """Renders one streamed message, block by block as each completes."""

    def __init__(self, renderer: MessageRenderer) -> None:
        self._renderer = renderer
        self._text = ''
        self._rendered_length = 0

    @property
    def text(self) -> str:
        return self._text

    def feed(self, delta: str) -> tuple[str, int] | None:
        """Append `delta`; return the HTML of the complete blocks and their source length
        in UTF-16 code units, or None when no further block has completed."""
        self._text += delta
        line_count = self._renderer.count_complete_lines(self._text)
        prefix = ''.join(self._text.splitlines(keepends=True)[:line_count])

        if len(prefix) <= self._rendered_length:
            return None

        self._rendered_length = len(prefix)
        return self._renderer.render(prefix), utf16_length(prefix)

    def finish(self) -> tuple[str, int]:
        """Return the HTML of the whole message and its source length in UTF-16 code units."""
        self._rendered_length = len(self._text)
        return self._renderer.render(self._text), utf16_length(self._text)
