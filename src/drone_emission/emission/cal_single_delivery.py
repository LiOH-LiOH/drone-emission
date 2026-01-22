import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EbikeParams:
    """电动自行车参数"""

    # 生产阶段排放
    e_body: float = 5.65  # 车身排放因子 (kg CO2e/kg)
    e_battery: float = 107  # 电池排放因子 (kg CO2e/kWh)
    e_body_dispose: float = 2.88  # 车身处置排放因子 (kg CO2e/kg)
    e_battery_dispose: float = 20.35  # 电池处置排放因子 (kg CO2e/kWh)

    # 物理参数
    w_body: float = 55  # 车身重量 (kg)
    w_battery: float = 30  # 电池重量 (6*5 kg)
    n_battery: int = 5  # 电池数量
    v_battery: float = 0.24  # 单块电池容量 (kWh) = 60V * 20Ah / 1000 * (1/5)

    # 运行参数
    velocity: float = 30  # 速度 (km/h)
    e_use: float = 0.6  # 使用阶段排放因子 (kg CO2e/kWh)

    # 寿命与数量
    lifespan: int = 4  # 使用寿命 (年)
    battery_lifespan: int = 2  # 电池寿命 (年)
    n_ebikes: int = 100  # 电动自行车数量

    # 能耗模型参数 (Table S5)
    g: float = 9.81  # 重力加速度 (m/s²)
    C_d: float = 0.65  # 空气阻力系数
    C_rr: float = 0.008  # 滚动阻力系数
    rho: float = 1.225  # 空气密度 (kg/m³)
    A: float = 1.0  # 迎风面积 (m²)
    grade: float = 0.015  # 道路坡度
    n_acc: int = 1  # 加速频率
    P_aux: float = 0.01  # 辅助设备功率 (kW)


@dataclass
class DroneParams:
    """无人机参数"""

    # 生产阶段排放
    e_body: float = 9.3  # 机身排放因子 (kg CO2e/kg)
    e_battery: float = 141  # 电池排放因子 (kg CO2e/kWh)
    e_body_dispose: float = 3.0  # 机身处置排放因子 (kg CO2e/kg)
    e_battery_dispose: float = 20.35  # 电池处置排放因子 (kg CO2e/kWh)

    # 物理参数
    w_body: float = 10.1  # 机身重量 (kg)
    w_battery: float = 3.0  # 电池总重量 (1.5*2 kg)
    n_battery: int = 4  # 电池数量
    v_battery: float = 0.389  # 单块电池容量 (kWh)

    # 运行参数
    v_level: float = 60  # 水平飞行速度 (km/h)
    v_vertical: float = 20  # 垂直起降速度 (km/h)
    h_takeoff: float = 100  # 起飞高度 (m)
    e_use: float = 0.6  # 使用阶段排放因子 (kg CO2e/kWh)

    # 寿命与数量
    lifespan: int = 4  # 使用寿命 (年)
    battery_lifespan: int = 2  # 电池寿命 (年)
    n_drones: int = 100  # 无人机数量

    # 能耗模型参数 (Table S5)
    g: float = 9.81  # 重力加速度 (m/s²)
    rho: float = 1.225  # 空气密度 (kg/m³)
    C_d: float = 0.65  # 空气阻力系数
    A: float = 0.15  # 迎风面积 (m²)
    r_rotor: float = 0.35  # 旋翼半径 (m)
    n_rotors: int = 8  # 旋翼数量
    n_blades: int = 5  # 每个旋翼叶片数
    c_bar: float = 0.1  # 平均弦长 (m)
    C_l: float = 0.4  # 升力系数
    C_d_blade: float = 0.075  # 叶片阻力系数
    k_lift: float = 1.15  # 升力裕度系数
    P_aux: float = 0.01  # 辅助设备功率 (kW)
    hover_time: float = 30  # 每次飞行悬停时间 (秒)


@dataclass
class CabinetParams:
    """取餐柜参数"""

    power_ebike: float = 0.2  # 普通取餐柜功率 (kW)
    power_drone: float = 0.4  # 无人机取餐柜功率 (kW)
    e_use: float = 0.6  # 使用阶段排放因子 (kg CO2e/kWh)
    e_solar: float = 0.0847  # 太阳能排放因子 (kg CO2e/kWh)
    operating_hours: int = 24  # 每日运行时间 (小时)
    n_cabinets: int = 120  # 取餐柜数量
    ghg_cabinet_ebike: float = 200  # 普通柜GHG (kg CO2e)
    ghg_cabinet_drone: float = 500  # 无人机柜GHG (kg CO2e)
    lifespan: int = 5  # 使用寿命 (年)


def create_ebike_params(exp_config: Dict = None) -> EbikeParams:
    """Create EbikeParams with config overrides from experiment.toml.

    Args:
        exp_config: Experiment config dict from load_config()

    Returns:
        EbikeParams with values from config or defaults
    """
    if exp_config is None:
        return EbikeParams()

    emission_params = exp_config.get("emission_params", {})
    ebike_config = emission_params.get("ebike", {})

    return EbikeParams(
        velocity=ebike_config.get("velocity", 30),
        e_use=emission_params.get("e_use", 0.6),
    )


def create_drone_params(exp_config: Dict = None) -> DroneParams:
    """Create DroneParams with config overrides from experiment.toml.

    Args:
        exp_config: Experiment config dict from load_config()

    Returns:
        DroneParams with values from config or defaults
    """
    if exp_config is None:
        return DroneParams()

    emission_params = exp_config.get("emission_params", {})
    drone_config = emission_params.get("drone", {})

    return DroneParams(
        v_level=drone_config.get("v_level", 60),
        v_vertical=drone_config.get("v_vertical", 20),
        h_takeoff=drone_config.get("h_takeoff", 100),
        hover_time=drone_config.get("hover_time", 30),
        e_use=emission_params.get("e_use", 0.6),
    )


def create_cabinet_params(exp_config: Dict = None) -> CabinetParams:
    """Create CabinetParams with config overrides from experiment.toml.

    Args:
        exp_config: Experiment config dict from load_config()

    Returns:
        CabinetParams with values from config or defaults
    """
    if exp_config is None:
        return CabinetParams()

    emission_params = exp_config.get("emission_params", {})

    return CabinetParams(
        e_use=emission_params.get("e_use", 0.6),
    )


def get_emission_params(exp_config: Dict = None) -> Dict:
    """Get emission parameters from config.

    Args:
        exp_config: Experiment config dict from load_config()

    Returns:
        Dict with payload_kg, rider_weight, and other common params
    """
    if exp_config is None:
        return {
            "payload_kg": 1.0,
            "rider_weight": 70,
            "e_use": 0.6,
        }

    emission_params = exp_config.get("emission_params", {})
    return {
        "payload_kg": emission_params.get("payload_kg", 1.0),
        "rider_weight": emission_params.get("rider_weight", 70),
        "e_use": emission_params.get("e_use", 0.6),
    }


class EbikeEnergyModel:
    """电动自行车能耗模型"""

    def __init__(self, params: EbikeParams = None):
        self.p = params or EbikeParams()
        self._power_cache: Dict[tuple, Tuple[float, Dict]] = {}

    def calculate_power(
        self, total_weight_kg: float, velocity_ms: float
    ) -> Tuple[float, Dict]:
        """
        计算电动自行车功率 (kW)（带缓存）
        公式来自 Table S6 公式 (1)

        P_ebike = m*g*C_rr*v + 0.5*rho*C_d*A*v³ + m*g*grade*v + n_acc*0.504/(2*3.6³*v)
        """
        # 缓存键
        cache_key = (round(total_weight_kg, 2), round(velocity_ms, 4))
        if cache_key in self._power_cache:
            return self._power_cache[cache_key]

        m_t = total_weight_kg / 1000  # 转换为吨
        v_ms = velocity_ms  # 速度 (m/s)
        v_kmh = v_ms * 3.6  # 速度 (km/h)
        g = self.p.g

        # 滚动阻力功率
        P_rolling = m_t * g * self.p.C_rr * v_ms  # kW

        # 空气阻力功率
        P_air = self.p.rho * self.p.C_d * self.p.A * (v_ms**3) / 2000  # kW

        # 爬坡功率
        P_grade = m_t * g * self.p.grade * v_ms  # kW

        # 加速功率 (简化)
        P_acc = (self.p.n_acc * 0.504 * v_kmh**3 * m_t) / (2 * 1000 * 3.6)  # kW

        P_total = P_rolling + P_air + P_grade + P_acc
        result = (
            P_total,
            {
                "P_rolling": P_rolling,
                "P_air": P_air,
                "P_grade": P_grade,
                "P_acc": P_acc,
            },
        )

        # print(
        #     f"Power breakdown: P_rolling={P_rolling:.2f} kW, P_air={P_air:.2f} kW, P_grade={P_grade:.2f} kW, P_acc={P_acc:.2f} kW"
        # )

        self._power_cache[cache_key] = result
        return result

    def calculate_energy_consumption(
        self, distance_km: float, payload_kg: float = 1.0, rider_weight: float = 70
    ) -> Dict:
        """
        计算单次配送能耗

        参数:
            distance_km: 配送距离 (km)
            payload_kg: 货物重量 (kg)
            rider_weight: 骑手重量 (kg)

        返回:
            能耗详情字典
        """
        # 总重量 = 车身 + 电池 + 骑手 + 货物
        total_weight_go = self.p.w_body + self.p.w_battery + rider_weight + payload_kg
        total_weight_return = self.p.w_body + self.p.w_battery + rider_weight

        # 速度转换: km/h -> m/s
        velocity_ms = self.p.velocity / 3.6

        # 计算功率
        P_total_go, power_breakdown_go = self.calculate_power(
            total_weight_go, velocity_ms
        )
        P_total_return, power_breakdown_return = self.calculate_power(
            total_weight_return, velocity_ms
        )

        # 配送时间 (小时)
        time_hours = distance_km / self.p.velocity

        # 能耗 (kWh)
        energy_kwh = P_total_go * time_hours + P_total_return * time_hours

        return {
            "distance_km": distance_km,
            "payload_kg": payload_kg,
            "velocity_kmh": self.p.velocity,
            "power_kw_go": P_total_go,
            "power_breakdown_go": power_breakdown_go,
            "power_kw_return": P_total_return,
            "power_breakdown_return": power_breakdown_return,
            "time_hours": time_hours,
            "time_minutes": time_hours * 60,
            "energy_kwh": energy_kwh,
        }


class DroneEnergyModel:
    """无人机能耗模型"""

    def __init__(self, params: DroneParams = None):
        self.p = params or DroneParams()
        self._power_cache: Dict[tuple, Tuple[float, Dict]] = {}

    def calculate_rotor_disc_area(self) -> float:
        """计算旋翼盘面积"""
        return math.pi * self.p.r_rotor**2

    def calculate_blade_tip_speed(self) -> float:
        """
        计算叶尖速度 (公式4)
        U = sqrt(6 * m * g / (rho * n_rotors * n_blades * c_bar * C_l * r_rotor))
        """
        m = self.p.w_body + self.p.w_battery  # 空载重量
        U = math.sqrt(
            6
            * m
            * self.p.g
            / (
                self.p.rho
                * self.p.n_rotors
                * self.p.n_blades
                * self.p.c_bar
                * self.p.C_l
                * self.p.r_rotor
            )
        )
        return U

    def calculate_solidity(self) -> float:
        """
        计算实度 (公式5)
        sigma = n_blades * c_bar / (pi * r_rotor)
        """
        return self.p.n_blades * self.p.c_bar / (math.pi * self.p.r_rotor)

    def _solve_induced_velocity(
        self, T: float, rho: float, R: float, v: float, alpha: float
    ) -> float:
        """
        求解诱导速度 w，方程为:
        T/(2*rho*R) = w * sqrt((w - v*sin(alpha))^2 + (v*cos(alpha))^2)

        两边平方后得到四次方程:
        w^4 - 2*a*w^3 + c*w^2 - k^2 = 0
        其中 k = T/(2*rho*R), a = v*sin(alpha), c = v^2

        特殊情况：
        - 悬停 (v=0): 方程简化为 w^4 = k^2, 解为 w = sqrt(k) = sqrt(T/(2*rho*R))
        """
        k = T / (2 * rho * R)

        # 悬停或极低速情况：直接使用解析解，避免数值不稳定
        if v < 1e-6:
            return math.sqrt(k)

        a = v * math.sin(alpha)
        c = v**2  # a^2 + b^2 = v^2*sin^2 + v^2*cos^2 = v^2

        # 四次方程系数: w^4 - 2*a*w^3 + c*w^2 - k^2 = 0
        # 使用 numpy 求解
        coeffs = [1, -2 * a, c, 0, -(k**2)]
        roots = np.roots(coeffs)

        # 筛选正实根 (诱导速度必须为正实数)
        real_positive_roots = [
            r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0
        ]

        if not real_positive_roots:
            # 如果没有正实根，使用悬停诱导速度作为近似
            # return math.sqrt(T / (2 * rho * R))
            raise ValueError(
                "No positive real root found for induced velocity equation."
            )

        # 返回最小的正实根
        return min(real_positive_roots)

    def calculate_flight_angle_power(
        self, total_weight: float, velocity_ms: float, flight_angle_deg: float
    ) -> Tuple[float, Dict]:
        """
        计算任意飞行角度的功率 (kW)（带缓存）

        参数:
            total_weight: 总重量 (kg)
            velocity_ms: 飞行速度 (m/s)
            flight_angle_deg: 飞行角度 (度)，正值为爬升，负值为下降
        """
        # 缓存键
        cache_key = (
            round(total_weight, 2),
            round(velocity_ms, 4),
            round(flight_angle_deg, 1),
        )
        if cache_key in self._power_cache:
            return self._power_cache[cache_key]

        m_kg = total_weight
        v_ms = velocity_ms
        g = self.p.g
        rho = self.p.rho

        flight_angle_rad = math.radians(flight_angle_deg)

        # 爬升功率 (W)
        P_climb = m_kg * g * v_ms * math.sin(flight_angle_rad)

        # 机身阻力 (N)
        D_body = 0.5 * rho * self.p.C_d * self.p.A * v_ms**2

        # 推力 (N)
        T = (
            math.sqrt((m_kg * g) ** 2 + D_body**2 + 2 * D_body * P_climb / v_ms)
            if v_ms > 0
            else m_kg * g
        )

        # 旋翼盘总面积
        R = self.calculate_rotor_disc_area() * self.p.n_rotors
        U = self.calculate_blade_tip_speed()
        sigma = self.calculate_solidity()

        # 攻角
        if v_ms > 0:
            alpha = math.atan(
                (-D_body - m_kg * g * math.sin(flight_angle_rad))
                / (m_kg * g * math.cos(flight_angle_rad))
            )
        else:
            alpha = 0

        # 求解诱导速度 w
        w = self._solve_induced_velocity(T, rho, R, v_ms, alpha)

        # 诱导功率 P_induced = T * w (kW)
        P_induced = T * w * self.p.k_lift / 1000

        # 空气阻力功率 P_drag (kW)
        P_drag = D_body * v_ms / 1000

        # 型阻功率 P_profile (kW)
        P_profile = (
            rho * R * U**3 * (1 + 3 * (v_ms / U) ** 2) * sigma * self.p.C_d_blade / 8000
        )

        P_climb_kw = P_climb / 1000  # 转换为 kW

        # 辅助设备功率
        P_aux = self.p.P_aux

        # 总功率
        P_total = P_induced + P_profile + P_drag + P_climb_kw + P_aux

        result = (
            P_total,
            {
                "P_induced": P_induced,
                "P_profile": P_profile,
                "P_drag": P_drag,
                "P_climb": P_climb_kw,
                "P_aux": P_aux,
                "induced_velocity": w,
                "thrust_N": T,
                "attack_angle_deg": math.degrees(alpha),
            },
        )

        self._power_cache[cache_key] = result
        return result

    def calculate_energy_consumption(
        self, distance_km: float, payload_kg: float = 1.0
    ) -> Dict:
        """
        计算单次配送完整能耗
        包括：起飞 -> 水平飞行 -> 降落 -> 悬停 -> 返程

        参数:
            distance_km: 配送距离 (km) - 单程直线距离
            payload_kg: 单个订单重量 (kg)
            n_orders: 配送订单数量
        """
        # 不同阶段的载重
        weight_empty = self.p.w_body + self.p.w_battery
        weight_loaded = weight_empty + payload_kg

        # 速度转换
        v_level_ms = self.p.v_level / 3.6
        v_vertical_ms = self.p.v_vertical / 3.6

        # ============ 去程 (满载) ============
        # 起飞
        P_takeoff, takeoff_breakdown = self.calculate_flight_angle_power(
            weight_loaded, v_vertical_ms, flight_angle_deg=90
        )
        t_takeoff = (self.p.h_takeoff / 1000) / self.p.v_vertical  # 小时
        E_takeoff = P_takeoff * t_takeoff

        # 水平飞行 (去程)
        P_level_out, level_out_breakdown = self.calculate_flight_angle_power(
            weight_loaded, v_level_ms, flight_angle_deg=0
        )
        t_level_out = distance_km / self.p.v_level  # 小时
        E_level_out = P_level_out * t_level_out

        # 降落
        P_landing_out, landing_out_breakdown = self.calculate_flight_angle_power(
            weight_loaded, v_vertical_ms, flight_angle_deg=-90
        )
        t_landing_out = (self.p.h_takeoff / 1000) / self.p.v_vertical
        E_landing_out = P_landing_out * t_landing_out

        # 悬停 (卸货)
        P_hover, hover_breakdown = self.calculate_flight_angle_power(
            weight_loaded, 0, flight_angle_deg=0
        )
        t_hover = self.p.hover_time / 3600  # 小时
        E_hover = P_hover * t_hover

        # ============ 返程 (空载) ============
        # 起飞
        P_takeoff_return, _ = self.calculate_flight_angle_power(
            weight_empty, v_vertical_ms, flight_angle_deg=90
        )
        E_takeoff_return = P_takeoff_return * t_takeoff

        # 水平飞行 (返程)
        P_level_return, level_return_breakdown = self.calculate_flight_angle_power(
            weight_empty, v_level_ms, flight_angle_deg=0
        )
        E_level_return = P_level_return * t_level_out

        # 降落
        P_landing_return, _ = self.calculate_flight_angle_power(
            weight_empty, v_vertical_ms, flight_angle_deg=-90
        )
        E_landing_return = P_landing_return * t_landing_out

        # 总能耗
        E_total = (
            E_takeoff
            + E_level_out
            + E_landing_out
            + E_hover
            + E_takeoff_return
            + E_level_return
            + E_landing_return
        )

        # 总时间
        t_total = 2 * (t_takeoff + t_level_out + t_landing_out) + t_hover

        return {
            "distance_km": distance_km,
            "payload_kg": payload_kg,
            "weight_loaded_kg": weight_loaded,
            "weight_empty_kg": weight_empty,
            # 功率详情
            "power_takeoff_kw": P_takeoff,
            "power_level_loaded_kw": P_level_out,
            "power_level_empty_kw": P_level_return,
            "power_landing_kw": P_landing_out,
            "power_hover_kw": P_hover,
            # 时间详情 (小时)
            "time_takeoff_h": t_takeoff,
            "time_level_h": t_level_out,
            "time_landing_h": t_landing_out,
            "time_hover_h": t_hover,
            "time_total_h": t_total,
            "time_total_min": t_total * 60,
            # 能耗详情 (kWh)
            "energy_takeoff_kwh": E_takeoff,
            "energy_level_out_kwh": E_level_out,
            "energy_landing_out_kwh": E_landing_out,
            "energy_hover_kwh": E_hover,
            "energy_takeoff_return_kwh": E_takeoff_return,
            "energy_level_return_kwh": E_level_return,
            "energy_landing_return_kwh": E_landing_return,
            "energy_total_kwh": E_total,
            # 功率分解
            "level_power_breakdown": level_out_breakdown,
        }


class LifeCycleEmissions:
    """生命周期碳排放计算"""

    def __init__(
        self,
        ebike_params: EbikeParams = None,
        drone_params: DroneParams = None,
        cabinet_params: CabinetParams = None,
    ):
        self.ebike_p = ebike_params or EbikeParams()
        self.drone_p = drone_params or DroneParams()
        self.cabinet_p = cabinet_params or CabinetParams()

        self.ebike_energy = EbikeEnergyModel(self.ebike_p)
        self.drone_energy = DroneEnergyModel(self.drone_p)

    def calculate_ebike_production_emissions(self) -> Dict:
        """
        计算电动自行车生产阶段排放
        公式 (1): CO2e_Epro+dis = e_Em * w_Em + e_Eb * V_Eb * n_Eb + e_Emd * w_Em + e_Ebd * V_Eb * n_Eb
        """
        p = self.ebike_p

        # 车身生产排放
        body_production = p.e_body * p.w_body

        # 电池生产排放
        total_battery_capacity = p.v_battery * p.n_battery  # kWh
        battery_production = p.e_battery * total_battery_capacity

        # 车身处置排放
        body_disposal = p.e_body_dispose * p.w_body

        # 电池处置排放
        battery_disposal = p.e_battery_dispose * total_battery_capacity

        total = body_production + battery_production + body_disposal + battery_disposal

        return {
            "body_production_kg": body_production,
            "battery_production_kg": battery_production,
            "body_disposal_kg": body_disposal,
            "battery_disposal_kg": battery_disposal,
            "total_kg": total,
            "total_battery_capacity_kwh": total_battery_capacity,
        }

    def calculate_drone_production_emissions(self) -> Dict:
        """
        计算无人机生产阶段排放
        公式 (2): CO2e_Upro+dis = e_Um * w_Um + e_Ub * V_Ub * n_Ub + e_Umd * w_Um + e_Ubd * V_Ub * n_Ub
        """
        p = self.drone_p

        # 机身生产排放
        body_production = p.e_body * p.w_body

        # 电池生产排放
        total_battery_capacity = p.v_battery * p.n_battery
        battery_production = p.e_battery * total_battery_capacity

        # 机身处置排放
        body_disposal = p.e_body_dispose * p.w_body

        # 电池处置排放
        battery_disposal = p.e_battery_dispose * total_battery_capacity

        total = body_production + battery_production + body_disposal + battery_disposal

        return {
            "body_production_kg": body_production,
            "battery_production_kg": battery_production,
            "body_disposal_kg": body_disposal,
            "battery_disposal_kg": battery_disposal,
            "total_kg": total,
            "total_battery_capacity_kwh": total_battery_capacity,
        }

    def calculate_ebike_use_phase_emissions(
        self, distance_km: float, payload_kg: float = 1.0, rider_weight: float = 70
    ) -> Dict:
        """
        计算电动自行车使用阶段排放
        公式 (3)

        参数:
            distance_km: 配送距离 (km)
            payload_kg: 货物重量 (kg)
            rider_weight: 骑手体重 (kg)
        """
        energy_result = self.ebike_energy.calculate_energy_consumption(
            distance_km, payload_kg, rider_weight
        )

        # 往返距离
        total_distance = distance_km * 2

        # 碳排放
        emissions_kg = energy_result["energy_kwh"] * self.ebike_p.e_use

        return {
            "total_energy_kwh": energy_result["energy_kwh"],
            "round_trip_distance_km": total_distance,
            "emissions_kg_co2e": emissions_kg,
            "emissions_g_co2e": emissions_kg * 1000,
            "time_total_min": energy_result["time_minutes"] * 2,
        }

    def calculate_drone_use_phase_emissions(
        self, distance_km: float, payload_kg: float = 1.0
    ) -> Dict:
        """
        计算无人机使用阶段排放
        公式 (4)
        """
        energy_result = self.drone_energy.calculate_energy_consumption(
            distance_km, payload_kg
        )

        # 碳排放
        emissions_kg = energy_result["energy_total_kwh"] * self.drone_p.e_use
        emissions_g = emissions_kg * 1000

        return {
            "total_energy_kwh": energy_result["energy_total_kwh"],
            "emissions_kg_co2e": emissions_kg,
            "emissions_g_co2e": emissions_g,
            "time_total_min": energy_result["time_total_min"],
            "energy_breakdown": {
                "takeoff": energy_result["energy_takeoff_kwh"],
                "level_out": energy_result["energy_level_out_kwh"],
                "landing_out": energy_result["energy_landing_out_kwh"],
                "hover": energy_result["energy_hover_kwh"],
                "takeoff_return": energy_result["energy_takeoff_return_kwh"],
                "level_return": energy_result["energy_level_return_kwh"],
                "landing_return": energy_result["energy_landing_return_kwh"],
            },
        }

    def calculate_cabinet_emissions_per_day(
        self, n_cabinets: int, is_drone: bool = False, use_solar: bool = False
    ) -> Dict:
        """
        计算取餐柜每日排放
        公式 (5): CO2e_CBuse = e_CBc * P_CB * t * n_CB
        """
        p = self.cabinet_p

        power = p.power_drone if is_drone else p.power_ebike
        e_factor = p.e_solar if use_solar else p.e_use

        # 每日能耗 (kWh)
        daily_energy = power * p.operating_hours * n_cabinets

        # 每日排放 (kg CO2e)
        daily_emissions = daily_energy * e_factor

        return {
            "power_kw": power,
            "n_cabinets": n_cabinets,
            "operating_hours": p.operating_hours,
            "daily_energy_kwh": daily_energy,
            "emission_factor": e_factor,
            "daily_emissions_kg": daily_emissions,
            "use_solar": use_solar,
        }

    def calculate_lifecycle_emissions_per_order(
        self,
        distance_km: float,
        n_orders_per_delivery: int,
        daily_orders: int,
        working_days: int = 300,
        vehicle_type: str = "ebike",
        use_solar: bool = False,
    ) -> Dict:
        """
        计算每订单生命周期碳排放

        参数:
            distance_km: 平均配送距离 (km)
            n_orders_per_delivery: 每次配送订单数
            daily_orders: 每日总订单数
            working_days: 年工作天数
            vehicle_type: 'ebike' 或 'drone'
            use_solar: 是否使用太阳能
        """
        annual_orders = daily_orders * working_days

        if vehicle_type == "ebike":
            # 生产阶段排放
            prod_emissions = self.calculate_ebike_production_emissions()

            # 年化生产排放 (考虑寿命和电池更换)
            annual_prod = (
                prod_emissions["body_production_kg"]
                + prod_emissions["body_disposal_kg"]
            ) / self.ebike_p.lifespan
            annual_battery = (
                prod_emissions["battery_production_kg"]
                + prod_emissions["battery_disposal_kg"]
            ) / self.ebike_p.battery_lifespan

            # 车队年化排放
            fleet_annual_prod = (annual_prod + annual_battery) * self.ebike_p.n_ebikes

            # 使用阶段排放
            use_emissions = self.calculate_ebike_use_phase_emissions(
                distance_km, 1.0 * n_orders_per_delivery
            )

            # 取餐柜排放
            cabinet_daily = self.calculate_cabinet_emissions_per_day(
                self.cabinet_p.n_cabinets, is_drone=False, use_solar=use_solar
            )

        else:  # drone
            # 生产阶段排放
            prod_emissions = self.calculate_drone_production_emissions()

            # 年化生产排放
            annual_prod = (
                prod_emissions["body_production_kg"]
                + prod_emissions["body_disposal_kg"]
            ) / self.drone_p.lifespan
            annual_battery = (
                prod_emissions["battery_production_kg"]
                + prod_emissions["battery_disposal_kg"]
            ) / self.drone_p.battery_lifespan

            # 车队年化排放
            fleet_annual_prod = (annual_prod + annual_battery) * self.drone_p.n_drones

            # 使用阶段排放 (无人机每次配送一个订单)
            use_emissions = self.calculate_drone_use_phase_emissions(distance_km, 1.0)

            # 取餐柜排放
            cabinet_daily = self.calculate_cabinet_emissions_per_day(
                self.cabinet_p.n_cabinets, is_drone=True, use_solar=use_solar
            )

        # 取餐柜年化生产排放
        ghg_per_cabinet = (
            self.cabinet_p.ghg_cabinet_drone
            if vehicle_type == "drone"
            else self.cabinet_p.ghg_cabinet_ebike
        )
        cabinet_prod_annual = (
            ghg_per_cabinet * self.cabinet_p.n_cabinets / self.cabinet_p.lifespan
        )

        # 年度总排放计算
        # 使用阶段: 每日订单数 / 每次配送订单数 = 每日配送次数
        daily_deliveries = daily_orders / n_orders_per_delivery
        daily_use_emissions = use_emissions["emissions_kg_co2e"] * daily_deliveries
        annual_use_emissions = daily_use_emissions * working_days

        # 取餐柜年度排放
        annual_cabinet_emissions = cabinet_daily["daily_emissions_kg"] * working_days

        # 总年度排放
        total_annual_emissions = (
            fleet_annual_prod
            + annual_use_emissions
            + annual_cabinet_emissions
            + cabinet_prod_annual
        )

        # 每订单排放
        emissions_per_order = (
            total_annual_emissions / annual_orders * 1000
        )  # g CO2e/order

        return {
            "vehicle_type": vehicle_type,
            "distance_km": distance_km,
            "n_orders_per_delivery": n_orders_per_delivery,
            "daily_orders": daily_orders,
            "annual_orders": annual_orders,
            # 各阶段年度排放 (kg CO2e)
            "fleet_production_annual_kg": fleet_annual_prod,
            "cabinet_production_annual_kg": cabinet_prod_annual,
            "use_phase_annual_kg": annual_use_emissions,
            "cabinet_operation_annual_kg": annual_cabinet_emissions,
            "total_annual_kg": total_annual_emissions,
            # 每订单排放 (g CO2e)
            "emissions_per_order_g": emissions_per_order,
            # 排放占比
            "production_share": (fleet_annual_prod + cabinet_prod_annual)
            / total_annual_emissions,
            "use_share": annual_use_emissions / total_annual_emissions,
            "cabinet_share": annual_cabinet_emissions / total_annual_emissions,
            # 单次配送详情
            "single_delivery_emissions_g": use_emissions["emissions_g_co2e"],
            "single_delivery_time_min": use_emissions.get("time_total_min"),
        }


def run_validation_tests():
    """运行验证测试，与论文结论对比"""

    print("=" * 80)
    print("无人机/电动自行车碳排放计算模型 - 验证测试")
    print("=" * 80)

    lca = LifeCycleEmissions()

    # ========================================================================
    # 测试1: 生产阶段排放
    # ========================================================================
    print("\n" + "=" * 60)
    print("测试1: 生产阶段排放计算")
    print("=" * 60)

    ebike_prod = lca.calculate_ebike_production_emissions()
    drone_prod = lca.calculate_drone_production_emissions()

    print("\n电动自行车生产排放:")
    print(f"  - 车身生产: {ebike_prod['body_production_kg']:.2f} kg CO2e")
    print(f"  - 电池生产: {ebike_prod['battery_production_kg']:.2f} kg CO2e")
    print(f"  - 车身处置: {ebike_prod['body_disposal_kg']:.2f} kg CO2e")
    print(f"  - 电池处置: {ebike_prod['battery_disposal_kg']:.2f} kg CO2e")
    print(f"  - 总排放: {ebike_prod['total_kg']:.2f} kg CO2e")
    print(f"  - 电池总容量: {ebike_prod['total_battery_capacity_kwh']:.2f} kWh")

    print("\n无人机生产排放:")
    print(f"  - 机身生产: {drone_prod['body_production_kg']:.2f} kg CO2e")
    print(f"  - 电池生产: {drone_prod['battery_production_kg']:.2f} kg CO2e")
    print(f"  - 机身处置: {drone_prod['body_disposal_kg']:.2f} kg CO2e")
    print(f"  - 电池处置: {drone_prod['battery_disposal_kg']:.2f} kg CO2e")
    print(f"  - 总排放: {drone_prod['total_kg']:.2f} kg CO2e")
    print(f"  - 电池总容量: {drone_prod['total_battery_capacity_kwh']:.2f} kWh")

    # ========================================================================
    # 测试2: 使用阶段能耗与排放
    # ========================================================================
    print("\n" + "=" * 60)
    print("测试2: 使用阶段能耗与排放 (不同距离)")
    print("=" * 60)

    distances = [0.5, 1, 2, 3]  # km

    r = 1.4  # 路径系数
    print(f"\n电动自行车 (单订单配送, r={r}):")
    print(f"{'距离(km)':<10} {'能耗(kWh)':<12} {'排放(g CO2e)':<15} {'时间(min)':<12}")
    print("-" * 50)
    for d in distances:
        d_route = d * r
        result = lca.calculate_ebike_use_phase_emissions(d_route, 1.0)
        print(
            f"{d_route:<10.2f} {result['total_energy_kwh']:<12.4f} {result['emissions_g_co2e']:<15.2f} {result['time_total_min']:<12.1f}"
        )

    print("\n无人机 (单订单配送):")
    print(f"{'距离(km)':<10} {'能耗(kWh)':<12} {'排放(g CO2e)':<15} {'时间(min)':<12}")
    print("-" * 50)
    for d in distances:
        result = lca.calculate_drone_use_phase_emissions(d, 1.0)
        print(
            f"{d:<10.2f} {result['total_energy_kwh']:<12.4f} {result['emissions_g_co2e']:<15.2f} {result['time_total_min']:<12.1f}"
        )

    # ========================================================================
    # 测试3: 论文核心结论验证
    # ========================================================================
    print("\n" + "=" * 60)
    print("测试3: 论文核心结论验证")
    print("=" * 60)

    print("\n论文结论: 使用阶段无人机GHG排放比电动自行车高25%-54%")
    print(f"验证 (5km距离, 1订单, r={r}):")

    ebike_use = lca.calculate_ebike_use_phase_emissions(5 * r, 1.0)
    drone_use = lca.calculate_drone_use_phase_emissions(5, 1.0)

    ratio = (drone_use["emissions_g_co2e"] / ebike_use["emissions_g_co2e"] - 1) * 100

    print(f"  - 电动自行车排放: {ebike_use['emissions_g_co2e']:.2f} g CO2e")
    print(f"  - 无人机排放: {drone_use['emissions_g_co2e']:.2f} g CO2e")
    print(f"  - 无人机相对增加: {ratio:.1f}%")

    return lca


def run_sensitivity_analysis(save_dir: str = None, exp_config: Dict = None, for_latex: bool = False):
    """
    运行敏感性分析

    参数:
        save_dir: 图片保存目录，如果为 None 则显示图形
        exp_config: 实验配置字典，从 experiment.toml 加载
        for_latex: 是否为 LaTeX 导出模式（PDF格式，无标题和信息框）
    """
    import os
    import sys

    # 添加 src/drone_emission 到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization.emission import (
        plot_distance_route_factor_heatmap,
        plot_energy_consumption_vs_speed,
        plot_sensitivity_analysis,
        plot_velocity_power_emission,
        plot_velocity_power_stacked,
    )

    print("\n" + "=" * 80)
    print("敏感性分析")
    print("=" * 80)

    # 从配置文件读取参数，使用默认值作为后备
    emission_cfg = get_emission_params(exp_config)
    payload_default = emission_cfg["payload_kg"]
    rider_weight_default = emission_cfg["rider_weight"]

    # 创建带配置的参数对象
    default_ebike_params = create_ebike_params(exp_config)
    default_drone_params = create_drone_params(exp_config)

    lca = LifeCycleEmissions(
        ebike_params=default_ebike_params,
        drone_params=default_drone_params,
    )
    r_default = 1.4  # 默认路径系数

    print(f"Config: payload={payload_default}kg, rider_weight={rider_weight_default}kg")
    print(f"E-bike: velocity={default_ebike_params.velocity}km/h")
    print(
        f"Drone: v_level={default_drone_params.v_level}km/h, h_takeoff={default_drone_params.h_takeoff}m"
    )

    # 确定文件扩展名
    ext = ".pdf" if for_latex else ".png"

    # ========================================================================
    # A2: 配送距离敏感性分析 (0.5 - 8 km)
    # ========================================================================
    print("\n" + "-" * 60)
    print("A2: 配送距离敏感性分析")
    print("-" * 60)

    distances = np.linspace(0.5, 8, 16)  # 0.5 到 8 km
    ebike_emissions_a2 = []
    drone_emissions_a2 = []

    for d in distances:
        ebike_result = lca.calculate_ebike_use_phase_emissions(
            d * r_default, payload_default
        )
        drone_result = lca.calculate_drone_use_phase_emissions(d, payload_default)
        ebike_emissions_a2.append(ebike_result["emissions_g_co2e"])
        drone_emissions_a2.append(drone_result["emissions_g_co2e"])

    print(f"{'距离(km)':<10} {'电动车(g)':<15} {'无人机(g)':<15} {'差异(%)':<10}")
    print("-" * 50)
    for i, d in enumerate(distances):
        diff = (drone_emissions_a2[i] / ebike_emissions_a2[i] - 1) * 100
        print(
            f"{d:<10.1f} {ebike_emissions_a2[i]:<15.2f} "
            f"{drone_emissions_a2[i]:<15.2f} {diff:<10.1f}"
        )

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/sensitivity_distance{ext}"
    plot_sensitivity_analysis(
        list(distances),
        ebike_emissions_a2,
        drone_emissions_a2,
        "Delivery Distance",
        "km",
        save_path=save_path,
        experiment_settings={
            "Payload": f"{payload_default:.2f} kg",
            "Route Factor": f"{r_default:.2f}",
            "E-bike Velocity": f"{default_ebike_params.velocity:.2f} km/h",
            "Drone Velocity": f"{default_drone_params.v_level:.2f} km/h",
        },
        for_latex=for_latex,
    )
    if save_path:
        print(f"\n图表已保存: {save_path}")

    # ========================================================================
    # A3: 路径系数敏感性分析 (1.2 - 2.0)
    # ========================================================================
    print("\n" + "-" * 60)
    print("A3: 路径系数敏感性分析")
    print("-" * 60)

    route_factors = np.linspace(1.2, 2.0, 9)
    d_fixed = 3.0  # 固定配送距离 3 km
    ebike_emissions_a3 = []
    drone_emissions_a3 = []

    for r in route_factors:
        ebike_result = lca.calculate_ebike_use_phase_emissions(
            d_fixed * r, payload_default
        )
        drone_result = lca.calculate_drone_use_phase_emissions(d_fixed, payload_default)
        ebike_emissions_a3.append(ebike_result["emissions_g_co2e"])
        drone_emissions_a3.append(drone_result["emissions_g_co2e"])

    print(f"{'路径系数':<10} {'电动车(g)':<15} {'无人机(g)':<15} {'差异(%)':<10}")
    print("-" * 50)
    for i, r in enumerate(route_factors):
        diff = (drone_emissions_a3[i] / ebike_emissions_a3[i] - 1) * 100
        print(
            f"{r:<10.2f} {ebike_emissions_a3[i]:<15.2f} "
            f"{drone_emissions_a3[i]:<15.2f} {diff:<10.1f}"
        )

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/sensitivity_route_factor{ext}"
    plot_sensitivity_analysis(
        list(route_factors),
        ebike_emissions_a3,
        drone_emissions_a3,
        "Route Factor",
        "",
        save_path=save_path,
        experiment_settings={
            "Distance": f"{d_fixed:.2f} km",
            "Payload": f"{payload_default:.2f} kg",
            "E-bike Velocity": f"{default_ebike_params.velocity:.2f} km/h",
            "Drone Velocity": f"{default_drone_params.v_level:.2f} km/h",
        },
        for_latex=for_latex,
    )
    if save_path:
        print(f"\n图表已保存: {save_path}")

    # ========================================================================
    # A7: 货物重量敏感性分析 (0.5 - 5 kg)
    # ========================================================================
    print("\n" + "-" * 60)
    print("A7: 货物重量敏感性分析")
    print("-" * 60)

    payloads = np.linspace(0.5, 5.0, 10)
    ebike_emissions_a7 = []
    drone_emissions_a7 = []

    for p in payloads:
        ebike_result = lca.calculate_ebike_use_phase_emissions(d_fixed * r_default, p)
        drone_result = lca.calculate_drone_use_phase_emissions(d_fixed, p)
        ebike_emissions_a7.append(ebike_result["emissions_g_co2e"])
        drone_emissions_a7.append(drone_result["emissions_g_co2e"])

    print(f"{'重量(kg)':<10} {'电动车(g)':<15} {'无人机(g)':<15} {'差异(%)':<10}")
    print("-" * 50)
    for i, p in enumerate(payloads):
        diff = (drone_emissions_a7[i] / ebike_emissions_a7[i] - 1) * 100
        print(
            f"{p:<10.1f} {ebike_emissions_a7[i]:<15.2f} "
            f"{drone_emissions_a7[i]:<15.2f} {diff:<10.1f}"
        )

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/sensitivity_payload{ext}"
    plot_sensitivity_analysis(
        list(payloads),
        ebike_emissions_a7,
        drone_emissions_a7,
        "Payload Weight",
        "kg",
        save_path=save_path,
        experiment_settings={
            "Distance": f"{d_fixed:.2f} km",
            "Route Factor": f"{r_default:.2f}",
            "E-bike Velocity": f"{default_ebike_params.velocity:.2f} km/h",
            "Drone Velocity": f"{default_drone_params.v_level:.2f} km/h",
        },
        for_latex=for_latex,
    )
    if save_path:
        print(f"\n图表已保存: {save_path}")

    # ========================================================================
    # 无人机水平速度敏感性分析 (40 - 80 km/h)
    # ========================================================================
    print("\n" + "-" * 60)
    print("无人机水平速度敏感性分析")
    print("-" * 60)

    drone_velocities = np.linspace(40, 80, 9)
    drone_powers = []
    drone_power_breakdowns = []
    drone_emissions_vel = []

    # 固定参数
    d_fixed = 3.0  # km
    payload_fixed = 1.0  # kg

    for v in drone_velocities:
        # 创建新的参数对象，修改水平速度
        drone_params = DroneParams(v_level=v)
        drone_energy = DroneEnergyModel(drone_params)

        # 计算功率 (水平飞行，满载)
        weight_loaded = drone_params.w_body + drone_params.w_battery + payload_fixed
        v_ms = v / 3.6
        power, breakdown = drone_energy.calculate_flight_angle_power(
            weight_loaded, v_ms, flight_angle_deg=0
        )
        drone_powers.append(power)
        drone_power_breakdowns.append(breakdown)

        # 计算排放
        energy_result = drone_energy.calculate_energy_consumption(
            d_fixed, payload_fixed
        )
        emissions_g = energy_result["energy_total_kwh"] * drone_params.e_use * 1000
        drone_emissions_vel.append(emissions_g)

    print(f"{'速度(km/h)':<12} {'功率(kW)':<12} {'排放(g CO2e)':<15}")
    print("-" * 40)
    for i, v in enumerate(drone_velocities):
        print(f"{v:<12.0f} {drone_powers[i]:<12.3f} {drone_emissions_vel[i]:<15.2f}")

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/sensitivity_drone_velocity{ext}"
    plot_velocity_power_emission(
        list(drone_velocities),
        drone_powers,
        drone_emissions_vel,
        vehicle_type="Drone",
        distance_info=f"distance={d_fixed}km",
        save_path=save_path,
        experiment_settings={
            "Distance": f"{d_fixed:.2f} km",
            "Payload": f"{payload_fixed:.2f} kg",
            "Takeoff Height": f"{default_drone_params.h_takeoff:.2f} m",
            "Hover Time": f"{default_drone_params.hover_time:.2f} s",
        },
        for_latex=for_latex,
    )
    if save_path:
        print(f"\n图表已保存: {save_path}")

    # ========================================================================
    # 电动车速度敏感性分析 (20 - 40 km/h)
    # ========================================================================
    print("\n" + "-" * 60)
    print("电动车速度敏感性分析")
    print("-" * 60)

    ebike_velocities = np.linspace(20, 40, 9)
    ebike_powers = []
    ebike_power_breakdowns = []
    ebike_emissions_vel = []
    ebike_energy_per_km = []  # Wh/km

    for v in ebike_velocities:
        # 创建新的参数对象，修改速度
        ebike_params = EbikeParams(velocity=v)
        ebike_energy = EbikeEnergyModel(ebike_params)

        # 计算功率
        total_weight = (
            ebike_params.w_body
            + ebike_params.w_battery
            + rider_weight_default
            + payload_fixed
        )
        v_ms = v / 3.6
        power, breakdown = ebike_energy.calculate_power(total_weight, v_ms)
        ebike_powers.append(power)
        ebike_power_breakdowns.append(breakdown)

        # 计算能耗 (1km距离)
        energy_result_1km = ebike_energy.calculate_energy_consumption(
            1.0, payload_fixed, rider_weight_default
        )
        energy_wh_per_km = energy_result_1km["energy_kwh"] * 1000
        ebike_energy_per_km.append(energy_wh_per_km)

        # 计算排放 (固定距离)
        energy_result = ebike_energy.calculate_energy_consumption(
            d_fixed * r_default, payload_fixed
        )
        emissions_g = energy_result["energy_kwh"] * ebike_params.e_use * 1000
        ebike_emissions_vel.append(emissions_g)

    print(
        f"{'速度(km/h)':<12} {'功率(kW)':<12} {'能耗(Wh/km)':<12} {'排放(g CO2e)':<15}"
    )
    print("-" * 55)
    for i, v in enumerate(ebike_velocities):
        print(
            f"{v:<12.0f} {ebike_powers[i]:<12.4f} "
            f"{ebike_energy_per_km[i]:<12.2f} {ebike_emissions_vel[i]:<15.2f}"
        )

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/sensitivity_ebike_velocity{ext}"
    plot_velocity_power_emission(
        list(ebike_velocities),
        ebike_powers,
        ebike_emissions_vel,
        vehicle_type="E-bike",
        distance_info=f"distance={d_fixed * r_default}km",
        save_path=save_path,
        experiment_settings={
            "Distance": f"{d_fixed * r_default:.2f} km (road)",
            "Payload": f"{payload_fixed:.2f} kg",
            "Route Factor": f"{r_default:.2f}",
            "Rider Weight": f"{rider_weight_default:.2f} kg",
        },
        for_latex=for_latex,
    )
    if save_path:
        print(f"\n图表已保存: {save_path}")

    # ========================================================================
    # 合并速度-功率堆叠柱状图 (无人机 + 电动车)
    # ========================================================================
    print("\n" + "-" * 60)
    print("合并速度-功率堆叠柱状图")
    print("-" * 60)

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/sensitivity_velocity_power_stacked{ext}"
    plot_velocity_power_stacked(
        list(drone_velocities),
        drone_power_breakdowns,
        list(ebike_velocities),
        ebike_power_breakdowns,
        save_path=save_path,
        experiment_settings={
            "Payload": f"{payload_fixed:.2f} kg",
            "Rider Weight": f"{rider_weight_default:.2f} kg",
            "Drone Body": f"{default_drone_params.w_body:.2f} kg",
            "E-bike Body": f"{default_ebike_params.w_body:.2f} kg",
        },
        for_latex=for_latex,
    )
    if save_path:
        print(f"图表已保存: {save_path}")

    # ========================================================================
    # 方案4(b): 配送距离 × 路径系数 热力图
    # ========================================================================
    print("\n" + "-" * 60)
    print("方案4(b): 配送距离 × 路径系数 二维热力图")
    print("-" * 60)

    distances_hm = np.linspace(0.5, 4, 30)
    route_factors_hm = np.linspace(1.0, 2.0, 20)

    emission_diff_matrix = np.zeros((len(route_factors_hm), len(distances_hm)))

    for i, r in enumerate(route_factors_hm):
        for j, d in enumerate(distances_hm):
            ebike_result = lca.calculate_ebike_use_phase_emissions(
                d * r, payload_default
            )
            drone_result = lca.calculate_drone_use_phase_emissions(d, payload_default)
            diff = (
                drone_result["emissions_g_co2e"] / ebike_result["emissions_g_co2e"] - 1
            ) * 100
            emission_diff_matrix[i, j] = diff

    print(f"矩阵尺寸: {emission_diff_matrix.shape}")
    print(
        f"排放差异范围: {emission_diff_matrix.min():.1f}% ~ {emission_diff_matrix.max():.1f}%"
    )

    save_path = None
    if save_dir:
        save_path = f"{save_dir}/sensitivity_distance_route_factor{ext}"
    plot_distance_route_factor_heatmap(
        list(distances_hm),
        list(route_factors_hm),
        emission_diff_matrix,
        save_path=save_path,
        experiment_settings={
            "Payload": f"{payload_default:.2f} kg",
            "E-bike Velocity": f"{default_ebike_params.velocity:.2f} km/h",
            "Drone Velocity": f"{default_drone_params.v_level:.2f} km/h",
        },
        for_latex=for_latex,
    )
    if save_path:
        print(f"\n图表已保存: {save_path}")

    print("\n" + "=" * 80)
    print("敏感性分析完成")
    print("=" * 80)


if __name__ == "__main__":
    import os
    import sys

    # 添加 src/drone_emission 到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.config import load_config

    # 获取项目根目录并加载配置
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    exp_config = load_config(os.path.join(project_root, "experiment.toml"))
    figures_dir = exp_config["figures_dir"]
    for_latex = exp_config.get("for_latex", False)

    lca = run_validation_tests()
    run_sensitivity_analysis(
        save_dir=os.path.join(figures_dir, "sensitivity"),
        exp_config=exp_config,
        for_latex=for_latex,
    )
