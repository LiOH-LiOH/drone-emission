# Figliozzi (2017) - Lifecycle modeling and assessment of UAV emissions

**Full Title:** Lifecycle modeling and assessment of unmanned aerial vehicles (Drones) CO2e emissions

**Author:** Miguel A. Figliozzi

**Journal:** Transportation Research Part D: Transport and Environment, Volume 57, pp. 251-261

**DOI:** 10.1016/j.trd.2017.09.011

---

## Research Focus
- **Comparison:** Drone vs Diesel Van / Electric Truck / Electric Van / Tricycle
- **Scope:** Generic analytical framework (USA context)
- **Method:** Life Cycle Assessment (LCA) - both utilization and vehicle phases
- **Verdict:** Mixed - UAVs efficient per-distance for small payloads, but lose advantage when customers grouped in routes

---

## Key Methodology

### Analytical Framework
- Physics-based energy consumption model for UAVs
- Considers lift-to-drag ratio, payload, vehicle mass, power transfer efficiency
- Two service configurations analyzed:
  - **One-to-one:** Single delivery per trip (UAV returns empty)
  - **One-to-many:** Multiple customers per route (ground vehicles only)

### LCA System Boundary
- **Utilization phase:** WTT (Well-to-Tank) + TTW (Tank-to-Wheel) for ground vehicles; GTB (Generation-to-Battery) + BTP (Battery-to-Propeller) for UAVs
- **Vehicle phase:** Raw material extraction, manufacturing, distribution, disposal/recycling
- Uses GREET model for emissions estimation
- eGRID database for electricity emissions

### Energy Model
UAV power for level flight:
```
p_l = (m_t + m_b + m_l) × g × d / (ϑ(s) × η_p)
```
Where ϑ(s) = lift-to-drag ratio, η_p = power transfer efficiency

---

## Key Findings

### One-to-One Service (Single Delivery)

| Metric | Diesel Van | UAV | Ratio |
|--------|------------|-----|-------|
| Energy per distance | 1016 Wh/km | 21.6 Wh/km | **47×** |
| Emissions per energy | 12.6 gCO2e/Wh | 0.6 gCO2e/Wh | 22.5× |
| Emissions per distance | 12.83 kgCO2e/km | 0.012 kgCO2e/km | **1056×** |
| Payload capacity | 1890 kg | 5 kg | 378× |
| Energy per distance-payload | 0.54 Wh/km·kg | 4.32 Wh/km·kg | 0.12× |

**Key insight:** UAVs highly efficient per-distance, but vans more efficient per distance-payload.

### One-to-Many Service (Route with Multiple Stops)

Breakeven points (n* = number of customers where efficiency equalizes):

| Comparison | Breakeven n* |
|------------|--------------|
| UAV vs Diesel Van | 58-362 customers |
| UAV vs Electric Truck | 42-214 customers |
| UAV vs Electric Van | 10-26 customers |
| UAV vs Electric Tricycle | **1.5-2.1 customers** |

**Critical finding:** Tricycles outcompete UAVs when just 2 customers can be grouped in a route.

### Vehicle Phase Emissions

| Parameter | UAV | Tricycle | Diesel Van |
|-----------|-----|----------|------------|
| Battery emissions (kg CO2e) | 435 | 306 | – |
| Vehicle emissions (kg CO2e) | 56 | 346 | 10,076 |
| Emissions per delivery (kg CO2e) | **0.16** | 0.02 | 0.03 |
| Equivalent travel distance (km) | 13.0 | 1.2 | 0.002 |
| As % of range | **52%** | 2.5% | 0% |

**Key insight:** UAV vehicle phase emissions can increase total emissions by up to 50%.

---

## Limitations Noted
- Assumes UAV cannot do multiple drops without returning to depot
- Limited UAV range (~25 km practical)
- No reliable data on UAV circuitousness factor (kc) in urban/rural areas
- Battery lifecycle assumptions (4 batteries over 3-year UAV life)

---

## Relevance to Your Work
- **OD Data:** Yes (uses continuous approximation formulas for route distances)
- **Scale:** Generic (applicable to any location with parameter adjustment)
- **Comparison vehicle:** Diesel van, electric truck/van, tricycle (not e-bike)
- **Key difference from your work:** Compares against tricycle, not e-bike; USA electricity grid; analytical rather than simulation-based

---

## Key Parameters Used

| Parameter | UAV (MD4-3000) | Diesel Van (RAM ProMaster) |
|-----------|----------------|---------------------------|
| Gross weight | 15.1 kg | 4060 kg |
| Tare weight | 10.1 kg | 2170 kg |
| Payload | 5.0 kg | 1890 kg |
| Energy consumption | 21.6 Wh/km | 1016 Wh/km |
| Range | 36 km | 695 km |
| L/D ratio (ϑ) | 3.5-5.0 | – |
| Grid emissions | 1.235 lbs CO2e/kWh | – |

---

## Key Equations

**Relative energy efficiency (one-to-one):**
```
ρ₁ᵉⁿ = (2 × kc × cf × fc × ϑ(s) × ηp × ηr) / (g × m × (1 + cm))
```

**Relative energy efficiency (one-to-many):**
```
ρₙᵉⁿ = (cf × fc × [2kc × d̄ + kl × √(nA)] × ϑ(s) × ηp × ηr) / (n × d̄ × g × m × (1 + cm))
```

---

## Contribution to Field
- **Foundational paper** establishing analytical framework for drone vs ground vehicle LCA
- First to compare UAVs against electric tricycles
- Introduced breakeven analysis for route configurations
- Highlighted importance of vehicle phase emissions for UAVs
