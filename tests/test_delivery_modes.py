import pytest

from drone_emission.emission import DELIVERY_MODES, lifecycle_emission_per_order
from drone_emission.emission.cal_single_delivery import (
    LifeCycleEmissions,
    create_drone_params,
    create_ebike_params,
)


EXP_CONFIG = {
    "emission_params": {
        "payload_kg": 1.0,
        "rider_weight": 70.0,
        "e_use": 0.6,
        "ebike": {"velocity": 30.0, "kappa": 0.7},
        "drone": {
            "v_level": 60.0,
            "v_vertical": 20.0,
            "h_takeoff": 100.0,
            "hover_time": 30.0,
        },
    }
}


def test_mode_contract_is_stable():
    assert DELIVERY_MODES == (
        "drone",
        "walking",
        "pedal_bicycle",
        "pedelec_ebike",
        "fuel_moped",
        "electric_moped",
        "fuel_car",
        "battery_electric_car",
    )


@pytest.mark.parametrize("mode", ["drone", "pedelec_ebike"])
def test_paper_modes_match_existing_lifecycle_model(mode):
    distance_km = 2.0
    grid_ef = 0.558
    ebike = create_ebike_params(EXP_CONFIG)
    drone = create_drone_params(EXP_CONFIG)
    ebike.e_use = grid_ef
    drone.e_use = grid_ef
    legacy = LifeCycleEmissions(ebike_params=ebike, drone_params=drone)
    vehicle_type = "drone" if mode == "drone" else "ebike"
    expected = legacy.calculate_lifecycle_emissions_per_order_simple(
        distance_km,
        n_orders_per_delivery=1,
        vehicle_type=vehicle_type,
    )["total_per_order_g"]

    observed = lifecycle_emission_per_order(
        mode,
        distance_km,
        exp_config=EXP_CONFIG,
        grid_ef_kg_per_kwh=grid_ef,
    )
    assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mode", "fixed_g", "energy", "factor"),
    [
        ("pedal_bicycle", 100_357.6974652074 / 13_440.0, 0.0, 0.0),
        (
            "fuel_moped",
            391_272.1646562054 / 49_000.0,
            0.6493506493506495,
            0.08901666666666667,
        ),
        (
            "electric_moped",
            480_144.80662530754 / 49_000.0,
            0.03501400560224091,
            0.558,
        ),
        (
            "fuel_car",
            6_496_824.912821 / 181_500.0,
            2.277184,
            0.08901666666666667,
        ),
        (
            "battery_electric_car",
            11_339_014.877200084 / 181_500.0,
            0.19034624761904761,
            0.558,
        ),
    ],
)
def test_itf_modes_use_lifetime_km_allocation(mode, fixed_g, energy, factor):
    distance_km = 3.25
    expected = 2 * distance_km * (fixed_g + energy * factor * 1000)
    observed = lifecycle_emission_per_order(
        mode,
        distance_km,
        exp_config=EXP_CONFIG,
        grid_ef_kg_per_kwh=0.558,
        fuel_wtw_ef_kg_per_mj=0.08901666666666667,
    )
    assert observed == pytest.approx(expected)


def test_walking_is_zero_and_ignores_irrelevant_factors():
    assert lifecycle_emission_per_order(
        "walking",
        5.0,
        exp_config=None,
        grid_ef_kg_per_kwh="not used",
        fuel_wtw_ef_kg_per_mj="not used",
    ) == 0.0


@pytest.mark.parametrize(
    ("mode", "missing_name"),
    [
        ("drone", "grid_ef_kg_per_kwh"),
        ("electric_moped", "grid_ef_kg_per_kwh"),
        ("fuel_moped", "fuel_wtw_ef_kg_per_mj"),
    ],
)
def test_relevant_energy_factor_is_required(mode, missing_name):
    with pytest.raises(ValueError, match=missing_name):
        lifecycle_emission_per_order(mode, 1.0, exp_config=EXP_CONFIG)


@pytest.mark.parametrize("distance", [-1.0, float("nan"), float("inf")])
def test_distance_must_be_finite_and_nonnegative(distance):
    with pytest.raises(ValueError, match="distance_km"):
        lifecycle_emission_per_order(
            "walking",
            distance,
            exp_config=None,
        )


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown delivery mode"):
        lifecycle_emission_per_order("motorcycle", 1.0, exp_config=EXP_CONFIG)


def test_ground_mode_is_linear_in_distance():
    kwargs = {
        "exp_config": EXP_CONFIG,
        "fuel_wtw_ef_kg_per_mj": 0.08901666666666667,
    }
    one_km = lifecycle_emission_per_order("fuel_car", 1.0, **kwargs)
    three_km = lifecycle_emission_per_order("fuel_car", 3.0, **kwargs)
    assert three_km == pytest.approx(3.0 * one_km)


def test_energy_factor_only_changes_relevant_mode():
    common = {
        "exp_config": EXP_CONFIG,
        "fuel_wtw_ef_kg_per_mj": 0.08901666666666667,
    }
    low_grid = lifecycle_emission_per_order(
        "electric_moped",
        2.0,
        grid_ef_kg_per_kwh=0.2,
        **common,
    )
    high_grid = lifecycle_emission_per_order(
        "electric_moped",
        2.0,
        grid_ef_kg_per_kwh=0.8,
        **common,
    )
    assert high_grid > low_grid

    fuel_low_grid = lifecycle_emission_per_order(
        "fuel_moped",
        2.0,
        grid_ef_kg_per_kwh=0.2,
        **common,
    )
    fuel_high_grid = lifecycle_emission_per_order(
        "fuel_moped",
        2.0,
        grid_ef_kg_per_kwh=0.8,
        **common,
    )
    assert fuel_high_grid == fuel_low_grid


@pytest.mark.parametrize("mode", ["drone", "pedelec_ebike"])
def test_paper_modes_require_explicit_experiment_config(mode):
    with pytest.raises(ValueError, match="exp_config"):
        lifecycle_emission_per_order(
            mode,
            1.0,
            exp_config=None,
            grid_ef_kg_per_kwh=0.558,
        )
