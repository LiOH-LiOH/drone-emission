from dataclasses import dataclass
from typing import Dict

import numpy as np

from emission.cal_single_delivery import (
    DroneEnergyModel,
    DroneParams,
    EbikeEnergyModel,
    EbikeParams,
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)


@dataclass
class RevenueParams:
    """Revenue parameters (CNY)"""

    base_fee: float = 3.0  # Base delivery fee (CNY)
    per_km_fee: float = 1.0  # Per-km surcharge (CNY/km)


@dataclass
class EbikeCostParams:
    """E-bike cost parameters (CNY)"""

    vehicle_cost: float = 3000  # Vehicle purchase cost (CNY)
    battery_cost: float = 500  # Single battery cost (CNY)
    rider_hourly_wage: float = 25.0  # Rider hourly wage (CNY/h)
    electricity_price: float = 0.6  # Electricity price (CNY/kWh)
    maintenance_per_year: float = 600  # Annual maintenance cost (CNY)
    insurance_per_year: float = 500  # Annual insurance cost (CNY)


@dataclass
class DroneCostParams:
    """Drone cost parameters (CNY)"""

    vehicle_cost: float = 100000  # Vehicle purchase cost (CNY)
    battery_cost: float = 3000  # Single battery cost (CNY)
    electricity_price: float = 0.6  # Electricity price (CNY/kWh)
    utm_annual_fee: float = 50000  # Annual UTM system fee for fleet (CNY)
    n_drones_fleet: int = 100  # Number of drones sharing UTM fee
    maintenance_per_year: float = 5000  # Annual maintenance cost (CNY)
    insurance_per_year: float = 3000  # Annual insurance cost (CNY)


def create_revenue_params(exp_config: Dict = None) -> RevenueParams:
    if exp_config is None:
        return RevenueParams()
    profit_cfg = exp_config.get("profit_params", {})
    rev_cfg = profit_cfg.get("revenue", {})
    return RevenueParams(
        base_fee=rev_cfg.get("base_fee", 3.0),
        per_km_fee=rev_cfg.get("per_km_fee", 1.0),
    )


def create_ebike_cost_params(exp_config: Dict = None) -> EbikeCostParams:
    if exp_config is None:
        return EbikeCostParams()
    profit_cfg = exp_config.get("profit_params", {})
    ebike_cfg = profit_cfg.get("ebike", {})
    return EbikeCostParams(
        vehicle_cost=ebike_cfg.get("vehicle_cost", 3000),
        battery_cost=ebike_cfg.get("battery_cost", 500),
        rider_hourly_wage=ebike_cfg.get("rider_hourly_wage", 25.0),
        electricity_price=profit_cfg.get("electricity_price", 0.6),
        maintenance_per_year=ebike_cfg.get("maintenance_per_year", 600),
        insurance_per_year=ebike_cfg.get("insurance_per_year", 500),
    )


def create_drone_cost_params(exp_config: Dict = None) -> DroneCostParams:
    if exp_config is None:
        return DroneCostParams()
    profit_cfg = exp_config.get("profit_params", {})
    drone_cfg = profit_cfg.get("drone", {})
    return DroneCostParams(
        vehicle_cost=drone_cfg.get("vehicle_cost", 100000),
        battery_cost=drone_cfg.get("battery_cost", 3000),
        electricity_price=profit_cfg.get("electricity_price", 0.6),
        utm_annual_fee=drone_cfg.get("utm_annual_fee", 50000),
        n_drones_fleet=drone_cfg.get("n_drones_fleet", 100),
        maintenance_per_year=drone_cfg.get("maintenance_per_year", 5000),
        insurance_per_year=drone_cfg.get("insurance_per_year", 3000),
    )


class ProfitAnalysis:
    """Per-order cost, revenue, and profit calculation for drone vs e-bike delivery."""

    def __init__(
        self,
        ebike_params: EbikeParams = None,
        drone_params: DroneParams = None,
        revenue_params: RevenueParams = None,
        ebike_cost_params: EbikeCostParams = None,
        drone_cost_params: DroneCostParams = None,
    ):
        self.ebike_p = ebike_params or EbikeParams()
        self.drone_p = drone_params or DroneParams()
        self.rev_p = revenue_params or RevenueParams()
        self.ebike_cost_p = ebike_cost_params or EbikeCostParams()
        self.drone_cost_p = drone_cost_params or DroneCostParams()

        self.ebike_energy = EbikeEnergyModel(self.ebike_p)
        self.drone_energy = DroneEnergyModel(self.drone_p)

    def calculate_revenue(self, distance_km: float) -> Dict:
        """Calculate revenue for a delivery at given distance."""
        revenue = self.rev_p.base_fee + self.rev_p.per_km_fee * distance_km
        return {
            "base_fee": self.rev_p.base_fee,
            "distance_fee": self.rev_p.per_km_fee * distance_km,
            "total_revenue_cny": revenue,
        }

    def calculate_ebike_cost_per_order(
        self,
        distance_km: float,
        payload_kg: float = 1.0,
        rider_weight: float = 70,
    ) -> Dict:
        """Calculate per-order cost breakdown for e-bike delivery.

        Uses the same amortization logic as
        LifeCycleEmissions.calculate_lifecycle_emissions_per_order_simple.
        """
        p = self.ebike_p
        cp = self.ebike_cost_p

        # Delivery throughput
        delivery_time_min = (2 * distance_km / p.velocity) * 60 + p.loading_time_min
        deliveries_per_day = p.working_hours_per_day * 60 / delivery_time_min
        deliveries_per_year = deliveries_per_day * p.working_days_per_year
        lifetime_deliveries = deliveries_per_year * p.lifespan

        # Vehicle amortization
        vehicle_amort = cp.vehicle_cost / lifetime_deliveries

        # Battery amortization (n_battery batteries, replaced every battery_lifespan years)
        n_replacements = p.lifespan / p.battery_lifespan
        battery_amort = (cp.battery_cost * p.n_battery * n_replacements) / lifetime_deliveries

        # Labor: rider wage for round-trip time
        labor = cp.rider_hourly_wage * (2 * distance_km / p.velocity)

        # Energy cost
        energy_result = self.ebike_energy.calculate_energy_consumption(
            distance_km, payload_kg, rider_weight
        )
        energy_kwh = energy_result["energy_kwh"]
        energy_cost = cp.electricity_price * energy_kwh

        # Maintenance & insurance per order
        maintenance = cp.maintenance_per_year / deliveries_per_year
        insurance = cp.insurance_per_year / deliveries_per_year

        total_cost = vehicle_amort + battery_amort + labor + energy_cost + maintenance + insurance

        return {
            "vehicle_amort": vehicle_amort,
            "battery_amort": battery_amort,
            "labor": labor,
            "energy_cost": energy_cost,
            "maintenance": maintenance,
            "insurance": insurance,
            "total_cost_cny": total_cost,
            "deliveries_per_year": deliveries_per_year,
            "energy_kwh": energy_kwh,
        }

    def calculate_drone_cost_per_order(
        self,
        distance_km: float,
        payload_kg: float = 1.0,
    ) -> Dict:
        """Calculate per-order cost breakdown for drone delivery.

        Drone has no operator — labor=0, but has UTM system cost.
        """
        p = self.drone_p
        cp = self.drone_cost_p

        # Delivery throughput
        delivery_time_min = (2 * distance_km / p.v_level) * 60 + p.loading_time_min
        deliveries_per_day = p.working_hours_per_day * 60 / delivery_time_min
        deliveries_per_year = deliveries_per_day * p.working_days_per_year
        lifetime_deliveries = deliveries_per_year * p.lifespan

        # Vehicle amortization
        vehicle_amort = cp.vehicle_cost / lifetime_deliveries

        # Battery amortization
        n_replacements = p.lifespan / p.battery_lifespan
        battery_amort = (cp.battery_cost * p.n_battery * n_replacements) / lifetime_deliveries

        # No labor for drone
        labor = 0.0

        # Energy cost
        energy_result = self.drone_energy.calculate_energy_consumption(
            distance_km, payload_kg
        )
        energy_kwh = energy_result["energy_total_kwh"]
        energy_cost = cp.electricity_price * energy_kwh

        # UTM cost per drone per order
        utm_cost = cp.utm_annual_fee / cp.n_drones_fleet / deliveries_per_year

        # Maintenance & insurance per order
        maintenance = cp.maintenance_per_year / deliveries_per_year
        insurance = cp.insurance_per_year / deliveries_per_year

        total_cost = vehicle_amort + battery_amort + labor + energy_cost + utm_cost + maintenance + insurance

        return {
            "vehicle_amort": vehicle_amort,
            "battery_amort": battery_amort,
            "labor": labor,
            "energy_cost": energy_cost,
            "utm_cost": utm_cost,
            "maintenance": maintenance,
            "insurance": insurance,
            "total_cost_cny": total_cost,
            "deliveries_per_year": deliveries_per_year,
            "energy_kwh": energy_kwh,
        }

    def calculate_profit_per_order(
        self,
        distance_km: float,
        vehicle_type: str = "ebike",
        payload_kg: float = 1.0,
        rider_weight: float = 70,
    ) -> Dict:
        """Calculate per-order profit (revenue - cost)."""
        rev = self.calculate_revenue(distance_km)

        if vehicle_type == "ebike":
            cost = self.calculate_ebike_cost_per_order(
                distance_km, payload_kg, rider_weight
            )
        else:
            cost = self.calculate_drone_cost_per_order(distance_km, payload_kg)

        profit = rev["total_revenue_cny"] - cost["total_cost_cny"]

        return {
            "vehicle_type": vehicle_type,
            "distance_km": distance_km,
            "revenue_cny": rev["total_revenue_cny"],
            "total_cost_cny": cost["total_cost_cny"],
            "profit_cny": profit,
            "cost_breakdown": cost,
        }

    def calculate_profit_for_od_df(
        self,
        distances_km: np.ndarray,
        vehicle_type: str = "ebike",
        payload_kg: float = 1.0,
        rider_weight: float = 70,
    ) -> np.ndarray:
        """Vectorized profit calculation for multiple OD pairs.

        Returns:
            Array of per-order profit (CNY).
        """
        results = np.zeros(len(distances_km))
        for i, d in enumerate(distances_km):
            r = self.calculate_profit_per_order(
                d, vehicle_type, payload_kg, rider_weight
            )
            results[i] = r["profit_cny"]
        return results


def run_profit_validation(exp_config: Dict = None):
    """Print cost/profit tables at sample distances for validation."""

    print("=" * 80)
    print("Profit Analysis - Validation")
    print("=" * 80)

    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    emission_cfg = get_emission_params(exp_config)
    payload_kg = emission_cfg["payload_kg"]
    rider_weight = emission_cfg["rider_weight"]

    pa = ProfitAnalysis(
        ebike_params=ebike_params,
        drone_params=drone_params,
        revenue_params=create_revenue_params(exp_config),
        ebike_cost_params=create_ebike_cost_params(exp_config),
        drone_cost_params=create_drone_cost_params(exp_config),
    )

    distances = [0.5, 1, 2, 3, 5]

    # Revenue table
    print("\n--- Revenue ---")
    print(f"{'Dist(km)':<10} {'Revenue(CNY)':<15}")
    print("-" * 25)
    for d in distances:
        rev = pa.calculate_revenue(d)
        print(f"{d:<10.1f} {rev['total_revenue_cny']:<15.2f}")

    # E-bike cost table
    print(f"\n--- E-bike Cost (payload={payload_kg}kg, rider={rider_weight}kg) ---")
    print(
        f"{'Dist(km)':<10} {'Vehicle':<10} {'Battery':<10} {'Labor':<10} "
        f"{'Energy':<10} {'Maint':<10} {'Insur':<10} {'Total':<10} {'Profit':<10}"
    )
    print("-" * 90)
    for d in distances:
        cost = pa.calculate_ebike_cost_per_order(d, payload_kg, rider_weight)
        rev = pa.calculate_revenue(d)
        profit = rev["total_revenue_cny"] - cost["total_cost_cny"]
        print(
            f"{d:<10.1f} {cost['vehicle_amort']:<10.4f} {cost['battery_amort']:<10.4f} "
            f"{cost['labor']:<10.4f} {cost['energy_cost']:<10.4f} "
            f"{cost['maintenance']:<10.4f} {cost['insurance']:<10.4f} "
            f"{cost['total_cost_cny']:<10.4f} {profit:<10.4f}"
        )

    # Drone cost table
    print(f"\n--- Drone Cost (payload={payload_kg}kg) ---")
    print(
        f"{'Dist(km)':<10} {'Vehicle':<10} {'Battery':<10} {'UTM':<10} "
        f"{'Energy':<10} {'Maint':<10} {'Insur':<10} {'Total':<10} {'Profit':<10}"
    )
    print("-" * 90)
    for d in distances:
        cost = pa.calculate_drone_cost_per_order(d, payload_kg)
        rev = pa.calculate_revenue(d)
        profit = rev["total_revenue_cny"] - cost["total_cost_cny"]
        print(
            f"{d:<10.1f} {cost['vehicle_amort']:<10.4f} {cost['battery_amort']:<10.4f} "
            f"{cost['utm_cost']:<10.4f} {cost['energy_cost']:<10.4f} "
            f"{cost['maintenance']:<10.4f} {cost['insurance']:<10.4f} "
            f"{cost['total_cost_cny']:<10.4f} {profit:<10.4f}"
        )

    # Comparison summary
    print("\n--- Profit Comparison ---")
    print(f"{'Dist(km)':<10} {'Ebike Profit':<15} {'Drone Profit':<15} {'Diff(D-E)':<15}")
    print("-" * 55)
    for d in distances:
        ebike_profit = pa.calculate_profit_per_order(d, "ebike", payload_kg, rider_weight)
        drone_profit = pa.calculate_profit_per_order(d, "drone", payload_kg)
        diff = drone_profit["profit_cny"] - ebike_profit["profit_cny"]
        print(
            f"{d:<10.1f} {ebike_profit['profit_cny']:<15.4f} "
            f"{drone_profit['profit_cny']:<15.4f} {diff:<15.4f}"
        )

    return pa


def run_profit_sensitivity(
    save_dir: str = None, exp_config: Dict = None, for_latex: bool = False
):
    """Sensitivity analysis varying distance, rider wage, and drone vehicle cost."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization.profit import (
        plot_cost_breakdown_by_distance,
        plot_profit_vs_distance,
    )

    print("\n" + "=" * 80)
    print("Profit Sensitivity Analysis")
    print("=" * 80)

    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    emission_cfg = get_emission_params(exp_config)
    payload_kg = emission_cfg["payload_kg"]
    rider_weight = emission_cfg["rider_weight"]

    pa = ProfitAnalysis(
        ebike_params=ebike_params,
        drone_params=drone_params,
        revenue_params=create_revenue_params(exp_config),
        ebike_cost_params=create_ebike_cost_params(exp_config),
        drone_cost_params=create_drone_cost_params(exp_config),
    )

    ext = ".pdf" if for_latex else ".png"

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Cost breakdown by distance
    print("\n--- Cost Breakdown by Distance ---")
    save_path = f"{save_dir}/cost_breakdown_by_distance{ext}" if save_dir else None
    plot_cost_breakdown_by_distance(
        pa, exp_config, save_path=save_path, for_latex=for_latex
    )
    if save_path:
        print(f"Saved: {save_path}")

    # 2. Profit vs distance curve (synthetic data)
    print("\n--- Profit vs Distance ---")
    distances = np.linspace(0.3, 6.0, 50)
    ebike_profits = []
    drone_profits = []
    revenues = []
    for d in distances:
        ep = pa.calculate_profit_per_order(d, "ebike", payload_kg, rider_weight)
        dp = pa.calculate_profit_per_order(d, "drone", payload_kg)
        rev = pa.calculate_revenue(d)
        ebike_profits.append(ep["profit_cny"])
        drone_profits.append(dp["profit_cny"])
        revenues.append(rev["total_revenue_cny"])

    import pandas as pd

    synth_df = pd.DataFrame(
        {
            "distance_km": distances,
            "ebike_profit_cny": ebike_profits,
            "drone_profit_cny": drone_profits,
            "revenue_cny": revenues,
        }
    )

    save_path = f"{save_dir}/profit_vs_distance{ext}" if save_dir else None
    plot_profit_vs_distance(synth_df, save_path=save_path, for_latex=for_latex)
    if save_path:
        print(f"Saved: {save_path}")

    # 3. Sensitivity: rider wage
    print("\n--- Sensitivity: Rider Hourly Wage ---")
    wages = [15, 20, 25, 30, 35]
    d_fixed = 3.0
    print(f"{'Wage(CNY/h)':<15} {'Ebike Cost':<15} {'Ebike Profit':<15}")
    print("-" * 45)
    for w in wages:
        pa_tmp = ProfitAnalysis(
            ebike_params=ebike_params,
            drone_params=drone_params,
            revenue_params=create_revenue_params(exp_config),
            ebike_cost_params=EbikeCostParams(
                **{**pa.ebike_cost_p.__dict__, "rider_hourly_wage": w}
            ),
            drone_cost_params=create_drone_cost_params(exp_config),
        )
        cost = pa_tmp.calculate_ebike_cost_per_order(d_fixed, payload_kg, rider_weight)
        rev = pa_tmp.calculate_revenue(d_fixed)
        profit = rev["total_revenue_cny"] - cost["total_cost_cny"]
        print(f"{w:<15} {cost['total_cost_cny']:<15.4f} {profit:<15.4f}")

    # 4. Sensitivity: drone vehicle cost
    print("\n--- Sensitivity: Drone Vehicle Cost ---")
    vehicle_costs = [50000, 75000, 100000, 150000, 200000]
    print(f"{'VehicleCost':<15} {'Drone Cost':<15} {'Drone Profit':<15}")
    print("-" * 45)
    for vc in vehicle_costs:
        pa_tmp = ProfitAnalysis(
            ebike_params=ebike_params,
            drone_params=drone_params,
            revenue_params=create_revenue_params(exp_config),
            ebike_cost_params=create_ebike_cost_params(exp_config),
            drone_cost_params=DroneCostParams(
                **{**pa.drone_cost_p.__dict__, "vehicle_cost": vc}
            ),
        )
        cost = pa_tmp.calculate_drone_cost_per_order(d_fixed, payload_kg)
        rev = pa_tmp.calculate_revenue(d_fixed)
        profit = rev["total_revenue_cny"] - cost["total_cost_cny"]
        print(f"{vc:<15} {cost['total_cost_cny']:<15.4f} {profit:<15.4f}")

    print("\n" + "=" * 80)
    print("Profit Sensitivity Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.config import load_config

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    exp_config = load_config(os.path.join(project_root, "experiment.toml"))
    figures_dir = exp_config["figures_dir"]
    for_latex = exp_config.get("for_latex", False)

    run_profit_validation(exp_config)
    run_profit_sensitivity(
        save_dir=os.path.join(figures_dir, "profit"),
        exp_config=exp_config,
        for_latex=for_latex,
    )
