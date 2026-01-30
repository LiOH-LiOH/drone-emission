# Stolaroff et al. (2018) - Energy use and life cycle GHG emissions of drones

**Full Title:** Energy use and life cycle greenhouse gas emissions of drones for commercial package delivery

**Authors:** Joshuah K. Stolaroff, Constantine Samaras, Emma R. O'Neill, Alia Lubers, Alexandra S. Mitchell, Daniel Ceperley

**Journal:** Nature Communications, Volume 9, Article 409

**DOI:** 10.1038/s41467-017-02411-5

---

## Research Focus
- **Comparison:** Drone vs Diesel Truck / Natural Gas Truck / Electric Truck / Gasoline Van / Electric Van / Personal Vehicle (gasoline & EV)
- **Scope:** USA (with regional grid variation across EPA e-GRID subregions)
- **Method:** Physics-based energy model calibrated with 1073 empirical flight segments + Life Cycle Assessment
- **Verdict:** Mixed - Small drones (0.5 kg) have 23-54% lower GHGs than diesel trucks; large drones (8 kg) only better with low-carbon electricity

---

## Key Methodology

### Empirical Data Collection
- **1073 flight segments** from quadcopter test flights
- Velocities: 2-12 m/s
- Payloads: 0-384 g
- Wind: 0-7 m/s (random orientation)
- Two test sites with different distance constraints

### Energy Model
Physics-based model accounting for:
- **Induced power** (lift against gravity)
- **Body drag** (empirically determined)
- **Battery/package drag**
- **Power efficiency** (η = 70% at higher speeds, 50% at low speeds)

**Total thrust equation:**
```
T = (m_body + m_batt + m_package) × g + F_drag
```

**Energy per distance:**
```
e = P / v
```

**Range calculation:**
```
R = (m_batt × s_batt × δ) / [(e_loaded + e_unloaded) × f]
```
Where δ = depth of discharge (80%), f = safety factor (1.2)

### LCA System Boundary
- Direct transportation emissions
- Upstream fuel/electricity emissions (using EPA e-GRID non-baseload factors)
- Battery manufacturing emissions (75 MJ, 4.7 kg CO2e per kg LiFePO4)
- **Warehouse energy** (0.35 kWh electricity + 0.93 MJ natural gas per package)

---

## Key Findings

### Drone Range and Energy Use

| Drone Type | Max Payload | Battery (base) | Range | Energy/Distance |
|------------|-------------|----------------|-------|-----------------|
| Quadcopter (3DR Iris) | 0.5 kg | 1 kg | **3.5 km** | 32 J/m (0.032 MJ/km) |
| Octocopter (Turbo Ace Infinity 9) | 8.1 kg | 10 kg | **4.2 km** | 266 J/m (0.266 MJ/km) |

**Key insight:** Maximum theoretical range ~5 km with current LiPo batteries (150 Wh/kg). Future batteries (250 Wh/kg) could extend range significantly.

### Energy Use Comparison (per km travel)

| Vehicle | Energy (MJ/km) |
|---------|----------------|
| Small drone | **0.032** |
| Large drone | **0.266** |
| Personal EV | ~0.6 |
| Personal car | ~2.5 |
| Gasoline van | ~3.1 |
| Electric truck | ~2.4 |
| Gasoline helicopter drone | ~6.5 |
| Diesel truck | ~7.3 |
| Natural gas truck | ~7.8 |

### GHG Emissions per Package (g CO2e)

| Delivery Mode | California | US Average | Missouri |
|---------------|------------|------------|----------|
| Small drone | **~550** | **~700** | **~850** |
| Large drone | **~1100** | **~1500** | **~1800** |
| Diesel truck | ~1200 | ~1200 | ~1200 |
| Natural gas truck | ~1100 | ~1100 | ~1100 |
| Electric truck | ~700 | ~1000 | ~1300 |
| Gasoline van | ~1500 | ~1500 | ~1500 |
| Personal vehicle | ~2100 | ~2100 | ~2100 |
| Personal EV | ~1100 | ~1700 | ~2200 |

**Key findings:**
- Small drones: **23-54% lower GHGs** than diesel truck (depending on grid)
- Large drones: **9% lower** (California) to **50% higher** (Missouri) than diesel truck
- Both drones better than personal vehicle pickup for single package

### Warehouse Requirements

**Critical insight:** Warehouse energy dominates drone emissions!

| City | Warehouses Needed (3.5 km range) |
|------|----------------------------------|
| San Francisco city | ~4 |
| SF Bay Area | ~112 |

- Warehouse energy: **0.35 kWh electricity + 0.93 MJ natural gas** per package
- Doubling drone range reduces warehouse nodes by **factor of 3.5**
- Warehouse emissions can account for **>50%** of total drone delivery emissions

### Sensitivity Analysis

**Most important parameters for small drones:**
1. Warehouse multiplier (# of additional warehouses)
2. Electricity intensity of warehousing
3. Carbon intensity of electricity
4. Number of hops for delivery

**Most important parameters for large drones:**
1. Flight electricity (equally important as warehousing)
2. Carbon intensity of electricity
3. Warehouse multiplier

**Most important parameters for trucks:**
1. Packages delivered per km (base: 0.94/km or 1.51/mile)
2. Electricity intensity of warehousing

### Fixed-Wing Comparison
- Fixed-wing drone (3D Robotics Aero-M): ~10 J/m
- **~50% more efficient** than quadcopter
- Potential for hybrid designs to further improve efficiency

---

## Key Parameters Used

| Parameter | Quadcopter | Octocopter |
|-----------|------------|------------|
| Model | 3D Robotics Iris | Turbo Ace Infinity 9 |
| Body mass | 1.28 kg | 6.9 kg |
| Max payload | 0.5 kg | 8.1 kg |
| Battery mass (base) | 1.0 kg | 10.0 kg |
| Battery energy density | 540 kJ/kg (150 Wh/kg) | 540 kJ/kg (150 Wh/kg) |
| Depth of discharge | 80% | 80% |
| Power efficiency (η) | 70% | 70% |
| Drag coefficient (body) | 1.5 | 1.5 |
| Flight velocity | 10 m/s | 10 m/s |
| Range | 3.5 km | 4.2 km |
| Charging efficiency | 81% (battery × charger) | 81% |
| T&D losses | 5% | 5% |
| Safety factor | 1.2 | 1.2 |

### Grid Carbon Intensity (g CO2e/kWh, non-baseload)

| Region | Direct | With Upstream |
|--------|--------|---------------|
| California (CAMX) | 437 | ~500 |
| US Average | 654 | ~750 |
| Missouri (SRMW) | 966 | ~1050 |

---

## Limitations Noted
- Copter-focused analysis (limited fixed-wing data)
- Uncertain warehouse requirements
- Does not include weather constraints on operations
- Battery cycle life assumptions uncertain (base: 300 cycles)
- Regional electricity mix varies over time
- Does not account for drone airways/airspace restrictions
- Truck efficiency improvements not captured

---

## Relevance to Your Work
- **OD Data:** No (uses generic delivery scenarios)
- **Scale:** National (USA with regional variation)
- **Comparison vehicle:** Trucks and vans (not e-bike specifically)
- **Key differences from your work:**
  - Very small payload focus (0.5 kg) - similar to food delivery
  - USA context with regional grid sensitivity
  - **Empirically calibrated** energy model (rare in literature)
  - Highlights **warehouse energy** as critical factor often overlooked
  - Does not compare against e-bike/bicycle

---

## Key Equations

**Power requirement (theoretical minimum for hover):**
```
P_min,hover = T^(3/2) / √(½πnD²ρ)
```

**Power with forward velocity:**
```
P_min = T × (v×sin(α) + v_i)
```

**Induced velocity:**
```
v_i = 2T / [πnD²ρ × √((v×cos(α))² + (v×sin(α) + v_i)²)]
```

**Actual power consumption:**
```
P = P_min / η
```

**Drag force:**
```
F_drag = Σ ½ρv²C_Di A_i
```

---

## Key Contribution to Field
- **First empirically calibrated** drone energy model in major journal
- Published in **Nature Communications** (high impact)
- Demonstrates **warehouse energy is critical** and often underestimated
- Provides **regional grid sensitivity** analysis for policy guidance
- Shows **small drones advantageous**, but large drones problematic
- Highlights **technology improvement paradox**: better batteries enable heavier loads → higher emissions
- Open methodology enables replication and extension
- Influential baseline for subsequent drone delivery LCA studies
