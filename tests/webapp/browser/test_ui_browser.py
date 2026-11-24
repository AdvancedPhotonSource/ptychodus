"""The web UI driven in a real browser against a served storage root.

`tsc` checks the UI's types; these check what it does when it runs. Each page boots,
shows what is seeded, sends the parameters its controls show, and reports a failure in
place rather than blanking. Every test ends with `ServedPage.assert_clean`, so an
uncaught exception, an unexpected error status or a URL no route declares fails it even
where the page looks right. Opt in with `uv run pytest -m browser`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from playwright.sync_api import Locator, Page, Response, expect

from ptychodus.api.processing import RunState, create_run_record
from webapp_ui_fixtures import ServedPage, path_of, query_of

pytestmark = [
    pytest.mark.browser,
    # pytest-base-url, which pytest-playwright requires, defines its session-scoped
    # base_url fixture as an instance method, which pytest 9 deprecates.
    pytest.mark.filterwarnings('ignore:Session-scoped fixtures defined as instance methods'),
]

OpenUI = Callable[[], ServedPage]


def _expect_image(page: Page, title: str) -> None:
    """Waits for the image panel to show a render captioned with `title`."""
    expect(page.locator('.image-wrap img')).to_be_visible()
    expect(page.locator('.caption')).to_contain_text(title)


def _node(page: Page, node_id: str) -> Locator:
    return page.locator(f'[data-node-id="{node_id}"]')


def _path_ends(suffix: str) -> Callable[[Response], bool]:
    return lambda response: path_of(response.url).endswith(suffix)


def test_every_page_boots_on_an_empty_storage_root(open_ui: OpenUI) -> None:
    ui = open_ui()
    ui.goto()
    page = ui.page
    buttons = page.locator('#nav button:not(.nav-logo)')
    labels = buttons.evaluate_all('(nodes) => nodes.map((n) => n.getAttribute("aria-label"))')
    assert labels, 'the nav bar offers no pages'

    for label in labels:
        button = page.get_by_role('button', name=label, exact=True)
        button.click()
        expect(button).to_have_class(re.compile(r'\bactive\b'))
        # A page with a listing of its own says the storage root is empty; a view of
        # products shows its count, zero.
        expect(
            page.locator('#content').get_by_text(
                re.compile(r'^No .+ in the storage root\.$|: 0 \[')
            )
        ).to_be_visible()

    ui.assert_clean()


def test_diffraction_page_shows_the_mean_and_each_pattern(
    open_ui: OpenUI, seed_diffraction: Callable[..., UUID]
) -> None:
    d = seed_diffraction()
    ui = open_ui()
    ui.goto('#diffraction')
    page = ui.page

    _node(page, f'diff:{d}').click()
    _expect_image(page, 'mean pattern')
    assert page.url.endswith(f'#diffraction/{d}')

    _node(page, f'diff:{d}:2').click()
    _expect_image(page, 'pattern 2')
    ui.assert_clean()


def test_product_page_describes_the_selected_product(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    p = seed_product(analyzable=True)
    ui = open_ui()
    ui.goto('#product')
    page = ui.page

    page.locator('table.data-table tbody tr').first.click()
    expect(page.locator('.detail-panel h2')).to_have_text('analyzable-product')
    expect(page.locator('.detail-panel dl.detail-list')).to_contain_text(str(p))
    assert page.url.endswith(f'#product/{p}')
    ui.assert_clean()


def test_listings_show_every_resource_past_one_page(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    # The list routes return at most 200 rows a request.
    for _ in range(201):
        seed_product()

    ui = open_ui()
    page = ui.page
    offsets = []
    page.on(
        'request',
        lambda request: (
            offsets.append(query_of(request.url)['offset'])
            if path_of(request.url).endswith('/api/v1/product')
            else None
        ),
    )
    ui.goto('#product')

    expect(page.locator('table.data-table tbody tr')).to_have_count(201)
    expect(page.locator('.info-label')).to_contain_text('Products: 201 [')
    assert offsets == ['0', '200']
    ui.assert_clean()


def test_probe_page_shows_all_modes_and_each_mode(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    p = seed_product(analyzable=True, modes=2)
    ui = open_ui()
    ui.goto('#probe')
    page = ui.page

    _node(page, f'probe:{p}').click()
    _expect_image(page, 'all modes')
    assert page.url.endswith(f'#product/{p}/probe')

    _node(page, f'probe:{p}:1').click()
    _expect_image(page, 'mode 2')
    ui.assert_clean()


def test_object_page_shows_the_object_and_each_layer(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    p = seed_product(analyzable=True, layers=2)
    ui = open_ui()
    ui.goto('#object')
    page = ui.page

    _node(page, f'object:{p}').click()
    _expect_image(page, 'object')
    assert page.url.endswith(f'#product/{p}/object')

    _node(page, f'object:{p}:1').click()
    _expect_image(page, 'layer 2')
    ui.assert_clean()


def test_positions_page_plots_every_ticked_product(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    a = seed_product(analyzable=True)
    b = seed_product(analyzable=True, position_shift_x_m=2e-9)
    ui = open_ui()
    ui.goto('#positions')
    page = ui.page
    boxes = page.locator('table.data-table td.check-col input')

    boxes.nth(0).check()
    expect(page.locator('.image-wrap img')).to_be_visible()

    with page.expect_response(_path_ends('/product/positions/image')) as plotted:
        boxes.nth(1).check()

    assert plotted.value.status == 200
    plotted_uuids = set(re.findall(r'uuid=([0-9a-f-]{36})', plotted.value.url))
    assert plotted_uuids == {str(a), str(b)}
    expect(page.locator('.image-wrap img')).to_be_visible()
    ui.assert_clean()


def test_fluorescence_page_renders_at_the_paired_products_pitch(
    open_ui: OpenUI,
    seed_product: Callable[..., UUID],
    seed_fluorescence: Callable[..., UUID],
) -> None:
    p = seed_product(analyzable=True)
    f = seed_fluorescence(derived_from=[{'kind': 'product', 'uuid': str(p)}], elements=['Fe', 'Cu'])
    ui = open_ui()
    ui.goto('#fluorescence')
    page = ui.page

    with page.expect_response(_path_ends(f'/fluorescence/{f}/image')) as summary:
        _node(page, f'flu:{f}').click()

    assert query_of(summary.value.url)['product_uuid'] == str(p)
    _expect_image(page, 'all elements')

    with page.expect_response(_path_ends(f'/fluorescence/{f}/elements/Cu/image')) as element:
        _node(page, f'flu:{f}:Cu').click()

    assert query_of(element.value.url)['product_uuid'] == str(p)
    _expect_image(page, 'Cu')
    ui.assert_clean()


def _choose(page: Page, control: str, value: str) -> dict[str, str]:
    """Chooses `value` in the ribbon's `control`, returning the render query it sends."""
    with page.expect_response(_path_ends('/probe/modes/image')) as rendered:
        page.locator(f'.image-ribbon select[title="{control}"]').select_option(value)

    assert rendered.value.status == 200, f'{control}={value}: {rendered.value.status}'
    expect(page.locator('.image-wrap img')).to_be_visible()
    return query_of(rendered.value.url)


def _option_values(page: Page, control: str) -> list[str]:
    select = page.locator(f'.image-ribbon select[title="{control}"]')
    return select.locator('option').evaluate_all('(nodes) => nodes.map((n) => n.value)')


def test_image_ribbon_sends_each_choice_it_offers(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    p = seed_product(analyzable=True)
    ui = open_ui()
    ui.goto('#probe')
    page = ui.page
    _node(page, f'probe:{p}').click()
    _expect_image(page, 'all modes')
    expect(page.locator('.image-ribbon')).to_be_visible()

    # The variants offered depend on the component: color models for the complex
    # renderer, a colormap for each other component.
    for component in _option_values(page, 'Array Component'):
        query = _choose(page, 'Array Component', component)

        if component == 'complex':
            assert 'component' not in query
            variant_parameter = 'color_model'
        else:
            assert query['component'] == component
            variant_parameter = 'colormap'

        assert variant_parameter in query
        variants = _option_values(page, 'Variant')

        for variant in {variants[0], variants[-1]} - {query[variant_parameter]}:
            assert _choose(page, 'Variant', variant)[variant_parameter] == variant

    for transform in _option_values(page, 'Transformation'):
        if transform != 'identity':
            assert _choose(page, 'Transformation', transform)['transform'] == transform

    # Editing either bound pins both, at the values the inputs show.
    with page.expect_response(_path_ends('/probe/modes/image')) as pinned:
        page.get_by_label('Display minimum').fill('0.125')
        page.get_by_label('Display minimum').press('Enter')

    query = query_of(pinned.value.url)
    assert float(query['value_min']) == 0.125
    assert 'value_max' in query

    with page.expect_response(_path_ends('/probe/modes/image')) as clipped:
        page.get_by_label('clip').check()

    assert query_of(clipped.value.url)['clip'] == 'true'
    ui.assert_clean()


@pytest.mark.parametrize('view', ['probe', 'object', 'positions'])
def test_metrics_panel_describes_and_compares_products(
    view: str, open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    a = seed_product(analyzable=True)
    b = seed_product(analyzable=True, position_shift_x_m=2e-9)
    pending = seed_product(
        data_file=None, run=create_run_record(RunState.RUNNING, command='ptychodus-reconstruct')
    )
    ui = open_ui()
    ui.goto(f'#product/{a}/{view}')
    page = ui.page
    panel = page.locator('section.metrics-panel')

    expect(panel.locator('dl.detail-list').first).to_be_visible()

    compare = page.get_by_label('Compare with')
    offered = compare.locator('option').evaluate_all('(nodes) => nodes.map((n) => n.value)')
    assert offered == ['', str(b)], f'{pending} is pending and {a} is the product itself'

    with page.expect_response(_path_ends(f'/product/{a}/{view}/comparison')) as compared:
        compare.select_option(str(b))

    assert compared.value.status == 200
    expect(panel.locator('dl.detail-list')).to_have_count(2)
    ui.assert_clean()


def test_deep_link_opens_the_view_with_its_product_selected(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    p = seed_product(analyzable=True)
    ui = open_ui()
    ui.goto(f'#product/{p}/probe')
    page = ui.page

    expect(page.get_by_role('button', name='Probe', exact=True)).to_have_class(
        re.compile(r'\bactive\b')
    )
    expect(_node(page, f'probe:{p}')).to_have_class(re.compile(r'\bselected\b'))
    _expect_image(page, 'all modes')
    ui.assert_clean()


def test_deep_link_to_an_unknown_product_says_so_and_still_lists(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    seed_product(analyzable=True)
    unknown = uuid4()
    ui = open_ui()
    ui.goto(f'#product/{unknown}')
    page = ui.page

    expect(page.locator('.detail-panel')).to_contain_text('404')
    expect(page.locator('table.data-table tbody tr')).to_have_count(1)
    ui.assert_clean(allow_status={404})


def test_malformed_hash_falls_back_to_the_first_page(open_ui: OpenUI) -> None:
    ui = open_ui()
    ui.goto('#positions/not/a/route')

    expect(ui.page.locator('#nav button.active')).to_have_attribute('aria-label', 'Diffraction')
    ui.assert_clean()


def test_run_without_data_shows_its_state_and_errors_in_place(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    p = seed_product(
        data_file=None, run=create_run_record(RunState.RUNNING, command='ptychodus-reconstruct')
    )
    ui = open_ui()
    ui.goto('#product')
    page = ui.page

    row = page.locator('table.data-table tbody tr.row-pending')
    expect(row).to_contain_text('waiting for first result')

    ui.goto('#probe')
    _node(page, f'probe:{p}').click()
    expect(page.locator('.image-wrap .error')).to_contain_text('no product data yet')
    expect(page.locator('.metrics-error')).to_contain_text('no product data yet')
    # The page stays mounted: the failure did not reach showFatalError.
    expect(page.locator('#content .page')).to_be_visible()
    ui.assert_clean(allow_status={404})


def test_download_links_fetch_each_resource_file(
    open_ui: OpenUI,
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
    seed_fluorescence: Callable[..., UUID],
) -> None:
    d = seed_diffraction()
    seed_product(analyzable=True)
    f = seed_fluorescence()
    ui = open_ui()
    page = ui.page
    selections = [
        ('#diffraction', _node(page, f'diff:{d}')),
        ('#product', page.locator('table.data-table tbody tr').first),
        ('#fluorescence', _node(page, f'flu:{f}')),
    ]

    for route, row in selections:
        ui.goto(route)
        row.click()
        href = page.locator('a.download-btn').get_attribute('href')
        assert href is not None
        response = page.request.get(f'{ui.base_url}{href}')
        assert response.status == 200, f'{route}: GET {href} -> {response.status}'

    ui.assert_clean()


def test_narrow_viewport_switches_between_listing_and_view(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    seed_product(analyzable=True)
    ui = open_ui()
    page = ui.page
    page.set_viewport_size({'width': 800, 'height': 900})
    ui.goto('#product')
    left = page.locator('.panel-left')
    right = page.locator('.panel-right')

    expect(page.locator('.tabs')).to_be_visible()
    expect(left).to_be_visible()
    expect(right).to_be_hidden()

    page.locator('table.data-table tbody tr').first.click()
    expect(right).to_be_visible()
    expect(left).to_be_hidden()

    page.get_by_role('button', name='Browse').click()
    expect(left).to_be_visible()
    ui.assert_clean()
