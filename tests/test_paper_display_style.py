import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from drone_emission.paper_display_style import apply_display_style


def test_grids_removed_but_reference_lines_and_3d_preserved():
    figure = plt.figure()
    axes = figure.add_subplot(121)
    axes.grid(True)
    axes.minorticks_on()
    axes.grid(True, which="minor")
    reference = axes.axhline(0)
    inset = axes.inset_axes([0.1, 0.1, 0.3, 0.3])
    inset.grid(True)
    spatial = figure.add_subplot(122, projection="3d")
    spatial.grid(True)
    apply_display_style(figure, "other.pdf")
    for target in (axes, inset):
        for axis in (target.xaxis, target.yaxis):
            assert not any(tick.gridline.get_visible() for tick in
                           axis.get_major_ticks() + axis.get_minor_ticks())
    assert reference.get_visible()
    assert spatial._draw_grid
    plt.close(figure)


def test_right_axis_starts_at_zero_without_changing_left_axis():
    figure, axes = plt.subplots()
    right = axes.twinx()
    axes.set_ylim(-10, 30)
    right.set_ylim(-2, 50)
    apply_display_style(figure, "province_total_ebike_lifecycle.pdf")
    assert right.get_ylim() == (0, 50)
    assert axes.get_ylim() == (-10, 30)
    plt.close(figure)


def test_edge_bars_fit_inside_shared_viewport_and_values_unchanged():
    figure, axes = plt.subplots()
    right = axes.twinx()
    bars = axes.bar([0.2, 3.5], [40, 75], width=0.156)
    axes.set_xlim(0.2, 3.5)
    axes.set_xticks([0.2, 2, 3.5])
    apply_display_style(figure, "city_distance_drone_share_emission_fit.pdf")
    left, limit = axes.get_xlim()
    assert left < bars[0].get_x()
    assert limit > bars[1].get_x() + bars[1].get_width()
    assert right.get_xlim() == pytest.approx((left, limit))
    assert [bar.get_height() for bar in bars] == [40, 75]
    assert list(axes.get_xticks()) == [0.2, 2, 3.5]
    plt.close(figure)
