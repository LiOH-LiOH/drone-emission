# Power Calculation Models for Drone and E-bike

This document describes the physics-based power consumption models used in the emission analysis for drone and e-bike delivery systems.

## 1. E-bike Power Model

The e-bike power model calculates the instantaneous mechanical power required to propel the vehicle at a given velocity. The total power is the sum of four components:

$$
P_{\text{ebike}} = P_{\text{rolling}} + P_{\text{air}} + P_{\text{grade}} + P_{\text{acc}}
$$

### Power Components

**Rolling Resistance Power:**
Energy dissipated due to tire deformation and friction between the tire and road surface. This force is proportional to the normal force (weight) and is relatively constant regardless of speed.
$$
P_{\text{rolling}} = m \cdot g \cdot C_{rr} \cdot v
$$

**Aerodynamic Drag Power:**
Energy required to overcome air resistance as the vehicle moves through the atmosphere. This force increases with the square of velocity, but power increases with the cube of velocity since $P = F \cdot v$.
$$
P_{\text{air}} = \frac{1}{2} \rho \cdot C_d \cdot A \cdot v^3
$$

**Grade Climbing Power:**
Energy needed to raise the vehicle's potential energy when traveling uphill. This is the work done against gravity on an inclined road surface.
$$
P_{\text{grade}} = m \cdot g \cdot \theta \cdot v
$$

**Acceleration Power:**
Energy required to change the vehicle's kinetic energy during speed variations. This accounts for stop-and-go traffic conditions typical in urban delivery scenarios.
$$
P_{\text{acc}} = \frac{n_{acc} \cdot 0.504 \cdot v_{km/h}^3 \cdot m_t}{2 \times 1000 \times 3.6}
$$

### Variable and Parameter Table

| Symbol | Description | Unit | Default Value |
|--------|-------------|------|---------------|
| $m$ | Total mass (vehicle + battery + rider + payload) | kg | ~156 kg |
| $g$ | Gravitational acceleration | m/s² | 9.81 |
| $C_{rr}$ | Rolling resistance coefficient | - | 0.008 |
| $v$ | Velocity | m/s | - |
| $v_{km/h}$ | Velocity | km/h | 30 |
| $\rho$ | Air density | kg/m³ | 1.225 |
| $C_d$ | Aerodynamic drag coefficient | - | 0.65 |
| $A$ | Frontal area | m² | 1.0 |
| $\theta$ | Road grade (slope) | - | 0.015 |
| $n_{acc}$ | Acceleration frequency | - | 1 |
| $m_t$ | Total mass | tonnes | - |
| $P_{aux}$ | Auxiliary power | kW | 0.01 |

---

## 2. Drone Power Model

The drone power model calculates the power required for multi-rotor flight at various flight angles. The total power consists of five components:

$$
P_{\text{drone}} = P_{\text{induced}} + P_{\text{profile}} + P_{\text{drag}} + P_{\text{climb}} + P_{\text{aux}}
$$

### Power Components

**Induced Power:**
The fundamental power required to generate lift by accelerating air downward through the rotor disc. This is the energy transferred to the air to create the upward thrust that counteracts gravity. It is the dominant power component during hover and low-speed flight.
$$
P_{\text{induced}} = T \cdot w \cdot k_{\text{lift}}
$$

where $T$ is thrust and $w$ is the induced velocity solved from momentum theory:
$$
\frac{T}{2 \rho R} = w \sqrt{(w - v \sin\alpha)^2 + (v \cos\alpha)^2}
$$

**Profile Power:**
Energy dissipated due to the aerodynamic drag of the rotor blades as they rotate through the air. This power is consumed by the viscous friction on the blade surfaces and increases significantly with blade tip speed.
$$
P_{\text{profile}} = \frac{\rho \cdot R \cdot U^3 \cdot \left(1 + 3\left(\frac{v}{U}\right)^2\right) \cdot \sigma \cdot C_{d,blade}}{8}
$$

**Fuselage Drag Power:**
Energy required to overcome the aerodynamic resistance of the drone's body (fuselage) as it moves through the air. Similar to vehicle air drag, this increases with the cube of forward velocity.
$$
P_{\text{drag}} = D_{\text{body}} \cdot v = \frac{1}{2} \rho \cdot C_d \cdot A \cdot v^3
$$

**Climb Power:**
Energy needed to increase the drone's gravitational potential energy during vertical ascent. This is the work done against gravity when gaining altitude, and becomes negative (energy recovery) during descent.
$$
P_{\text{climb}} = m \cdot g \cdot v \cdot \sin\alpha
$$

**Auxiliary Power:**
Constant power consumption for onboard electronics, sensors, flight controller, GPS, communication systems, and other avionics that operate regardless of flight conditions.
$$
P_{\text{aux}} = \text{constant}
$$

### Derived Parameters

**Blade Tip Speed:**
The velocity of the rotor blade tips, which determines the rotor's ability to generate thrust. This is derived from the requirement that the blades must generate sufficient lift to support the aircraft weight.
$$
U = \sqrt{\frac{6 \cdot m \cdot g}{\rho \cdot n_r \cdot n_b \cdot \bar{c} \cdot C_l \cdot r}}
$$

**Rotor Solidity:**
The ratio of total blade area to rotor disc area. Higher solidity means more blade surface area relative to the disc swept by the rotor, affecting thrust generation and profile drag.
$$
\sigma = \frac{n_b \cdot \bar{c}}{\pi \cdot r}
$$

**Total Rotor Disc Area:**
The combined swept area of all rotors, representing the total area through which air is accelerated to generate lift.
$$
R = n_r \cdot \pi \cdot r^2
$$

### Variable and Parameter Table

| Symbol | Description | Unit | Default Value |
|--------|-------------|------|---------------|
| $m$ | Total mass (body + battery + payload) | kg | ~14.1 kg |
| $g$ | Gravitational acceleration | m/s² | 9.81 |
| $v$ | Flight velocity | m/s | - |
| $v_{\text{level}}$ | Horizontal flight velocity | km/h | 60 |
| $v_{\text{vertical}}$ | Vertical flight velocity | km/h | 20 |
| $\alpha$ | Flight path angle | rad | - |
| $\rho$ | Air density | kg/m³ | 1.225 |
| $C_d$ | Fuselage drag coefficient | - | 0.65 |
| $A$ | Fuselage frontal area | m² | 0.15 |
| $r$ | Rotor radius | m | 0.35 |
| $n_r$ | Number of rotors | - | 8 |
| $n_b$ | Number of blades per rotor | - | 5 |
| $\bar{c}$ | Mean blade chord | m | 0.1 |
| $C_l$ | Blade lift coefficient | - | 0.4 |
| $C_{d,blade}$ | Blade drag coefficient | - | 0.075 |
| $k_{\text{lift}}$ | Lift margin factor | - | 1.15 |
| $T$ | Thrust force | N | - |
| $w$ | Induced velocity | m/s | - |
| $U$ | Blade tip speed | m/s | - |
| $\sigma$ | Rotor solidity | - | - |
| $R$ | Total rotor disc area | m² | - |
| $D_{\text{body}}$ | Fuselage drag force | N | - |
| $P_{\text{aux}}$ | Auxiliary power | kW | 0.01 |
| $h_{\text{takeoff}}$ | Takeoff/landing height | m | 100 |
| $t_{\text{hover}}$ | Hover time per delivery | s | 30 |

---

## 3. Power Component Comparison

The stacked bar chart visualization (`sensitivity_velocity_power_stacked.png`) shows how each power component varies with velocity:

### Drone (40-80 km/h horizontal flight)
- **Profile power** dominates (~70-80% of total)
- **Induced power** relatively constant (~10-15%)
- **Drag power** increases with $v^3$ (becomes significant at high speeds)
- **Climb power** = 0 for horizontal flight

### E-bike (20-40 km/h)
- **Acceleration power** is significant (~25-30%)
- **Air drag** increases with $v^3$ (dominant at high speeds)
- **Grade resistance** constant proportion (~20-25%)
- **Rolling resistance** relatively constant (~15-20%)

---

## References

- Energy consumption model parameters from Table S5-S6 in supplementary materials
- Implementation: `src/drone_emission/emission/cal_single_delivery.py`
