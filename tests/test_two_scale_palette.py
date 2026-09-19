from dataclasses import replace
from types import MappingProxyType

import numpy as np

from drone_emission.visualization.palette import configure_palette, colormap, load_palette


def test_only_two_ramps_and_all_named_colormaps_resolve_to_them():
    palette = configure_palette()
    assert set(palette.ramps) == {"sequential", "diverging"}
    for key, target in palette.colormaps.items():
        kind = target.removeprefix("@scale.")
        assert kind in palette.ramps
        samples = colormap(key)(np.linspace(0, 1, 257))[:, :3]
        np.testing.assert_array_equal(samples, palette.ramps[kind])


def test_approved_endpoints_and_exact_zero_colour():
    palette = configure_palette()
    from matplotlib.colors import to_hex
    seq = colormap("city_priorities.reduction")
    signed = colormap("contribution_break_even.delta")
    assert [to_hex(seq(v)) for v in (0., 1.)] == ["#4ccfb5", "#123f7a"]
    assert [to_hex(signed(v)) for v in (0., .5, 1.)] == ["#123f7a", "#edbe42", "#d64f4f"]
    for mode in ("mode.drone", "mode.pedelec_ebike", "mode.fuel_car"):
        assert palette.values[mode].startswith(("@base.", "@derived."))


def test_lookup_table_edits_invalidate_render_signature():
    palette = load_palette()
    usage = {"colors": [], "colormaps": ["city_priorities.reduction"], "alpha": []}
    ramps = {key: [list(row) for row in table] for key, table in palette.ramps.items()}
    ramps["sequential"][100][0] += .001
    changed = replace(palette, ramps=MappingProxyType(ramps))
    assert palette.signature_for(usage) != changed.signature_for(usage)


def test_gef_reference_and_cross_figure_mapping():
    from drone_emission.visualization.palette import gef_norm
    configure_palette()
    norm = gef_norm()
    np.testing.assert_allclose(norm([.05, .3, .8]), [0, .5, 1])
    values = [.05, .1073, .3, .5366, .7252, .8]
    reference = colormap("regions.gef")(norm(values))
    for key in ["city_gef_support.gef"]:
        np.testing.assert_array_equal(colormap(key)(gef_norm()(values)), reference)


def test_gef_colorbar_values_are_equidistant_in_both_orientations():
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from drone_emission.visualization.palette import gef_norm, linearize_colorbar
    configure_palette()
    for orientation in ["horizontal", "vertical"]:
        fig, ax = plt.subplots()
        cbar = fig.colorbar(ScalarMappable(norm=gef_norm(), cmap=colormap("regions.gef")),
                            ax=ax, orientation=orientation, ticks=[.05,.3,.55,.8])
        linearize_colorbar(cbar)
        np.testing.assert_array_equal(cbar.get_ticks(), [.05,.3,.55,.8])
        fig.canvas.draw()
        values = np.array([.05, .3, .55, .8])
        dimension = 0 if orientation == "horizontal" else 1
        xy = np.full((4, 2), .5)
        xy[:, dimension] = values
        pixels = cbar.ax.transData.transform(xy)[:, dimension]
        np.testing.assert_allclose(np.diff(pixels), np.repeat(np.diff(pixels)[0], 3))
        np.testing.assert_allclose((pixels[1]-pixels[0])/(pixels[-1]-pixels[0]), 1/3)
        plt.close(fig)


def test_surface_uses_sequential_scale_with_same_numeric_limits():
    from drone_emission.visualization.palette import gef_norm
    configure_palette()
    norm = gef_norm(sequential=True)
    np.testing.assert_allclose(norm([.05,.3,.55,.8]), [0,1/3,2/3,1])
    np.testing.assert_array_equal(colormap("contribution_break_even.gef")(np.linspace(0,1,257)),
                                  colormap("city_priorities.reduction")(np.linspace(0,1,257)))


def test_fit_colour_follows_group_median_and_is_darker():
    from drone_emission.visualization.palette import gef_norm, gef_fit_color
    configure_palette()
    for values in [[.1073,.1404,.1567], [.44,.56,.7252]]:
        base = colormap("city_gef_support.gef")(gef_norm()(np.median(values)))[:3]
        result = gef_fit_color(values)
        assert sum(result) < sum(base)
        np.testing.assert_allclose(result, .78*np.array(base)+.22*.2)
        np.testing.assert_array_equal(result, gef_fit_color(values[::-1]))


def test_gef_normalization_edits_invalidate_style_signature():
    palette = load_palette()
    changed = replace(palette, normalizations=MappingProxyType({"gef": [.05, .4, .8]}))
    usage = {"colors": [], "colormaps": ["regions.gef"], "alpha": []}
    assert palette.signature_for(usage) != changed.signature_for(usage)
