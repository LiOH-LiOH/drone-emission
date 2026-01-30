# Kirschstein (2020) - Comparison of energy demands of drone-based and ground-based parcel delivery

**Full Title:** Comparison of energy demands of drone-based and ground-based parcel delivery services

**Author:** Thomas Kirschstein

**Journal:** Transportation Research Part D: Transport and Environment, Volume 78, 102209

**DOI:** 10.1016/j.trd.2019.102209

---

## Research Focus
- **Comparison:** Drone vs Diesel Truck (DV) vs Electric Truck (EV)
- **Scope:** Generic (Berlin, Germany road network simulation)
- **Method:** Use-phase energy modeling (Well-to-Wheel analysis)
- **Verdict:** Yes (drones greener only in rural settings with low customer density and calm wind)

---

## Key Methodology

### Energy Consumption Models

**Diesel/Electric Trucks:**
Physics-based model accounting for:
- Rolling resistance (P^roll)
- Air drag (P^air)
- Grade resistance (P^grade)
- Inertia/acceleration (P^inert)

**UAV (Multicopter):**
Detailed model accounting for:
- Air drag (P^air)
- Lifting power (P^lift)
- Rotor profile drag (P^profile)
- Climbing power (P^climb)
- Internal electronics (P^int)

### Flight Pattern Modeled
```
altitude
    │     level flight              level flight
    │    ┌──────────┐              ┌──────────┐
    │   /│          │\  hovering  /│          │\
    │  / │          │ \    │     / │          │ \
    │ /  │          │  \   ▼    /  │          │  \
    └────┴──────────┴───────────┴──────────┴────── time
     takeoff                landing  takeoff
```

### Well-to-Wheel (WTW) Analysis
- **TTW (Tank-to-Wheel):** Direct energy consumption
- **WTT (Well-to-Tank):** Upstream energy for fuel/electricity production
- **WTT Efficiency:**
  - Diesel: ~90%
  - Electricity (Germany): ~50% (varies 35-95% depending on grid mix)

### Simulation Setup
- **Location:** Berlin road network
- **Customers per route:** 110-200 (100 stops)
- **Customer area radius:** 2-8 km
- **UAV radius of operation:** 9 km
- **Hovering time:** 5 min per delivery
- **28,800 simulation runs total**

---

## Key Findings

### Energy Consumption Comparison

| Scenario | DV vs EV | UAV vs EV | UAV vs DV |
|----------|----------|-----------|-----------|
| Low traffic, 2km radius | DV +40-50% | UAV +200-400% | UAV worse |
| High traffic, 2km radius | DV +80-90% | UAV +300-500% | UAV worse |
| Low traffic, 8km radius | DV +40-50% | UAV +50-100% | UAV comparable |
| High traffic, 8km radius | DV +80-90% | UAV similar | UAV better than DV |

**Key insight:** EVs always consume least energy. UAVs only competitive in rural settings.

### Effect of Environmental Conditions

**Wind Speed Impact on UAV:**
| Wind Condition | Mean Speed | Energy Increase |
|----------------|------------|-----------------|
| Low (calm) | 0 km/h | Baseline |
| Medium | 25 km/h | +50-100% |
| High | 45 km/h | +200-300% |

**Traffic Congestion Impact on Trucks:**
- Low congestion: Trucks travel at speed limits
- High congestion: 67% of speed limit, 2× acceleration frequency
- **Counterintuitively:** EVs are MORE efficient at low speeds (high congestion)

### Hovering Energy Penalty
- Delivery drones designed for cruising, NOT hovering
- 5 min hovering adds significant energy demand
- At 35 km/h wind: Battery capacity exceeded after just 4 min hovering
- Hovering severely limits radius of operation

### CO2e Emissions Analysis

**Breakeven CO2 emission coefficient (co^elec) for UAV vs DV:**

| Scenario | co^elec threshold |
|----------|-------------------|
| 2km radius, 2.0 cust/stop | UAV always worse |
| 8km radius, 2.0 cust/stop | < 0.3 kg CO2e/kWh |
| 8km radius, 1.1 cust/stop | < 0.5 kg CO2e/kWh |

**Context:** Germany's grid: ~0.5 kg CO2e/kWh; Coal power: ~1.0 kg CO2e/kWh; Renewables: ~0

**Key insight:** Even with renewable electricity, UAVs emit more CO2e than DVs in dense urban areas.

---

## Critical Parameters

### UAV (Amazon-style drone)
| Parameter | Current | Improved |
|-----------|---------|----------|
| Tare weight | 12 kg | 12 kg |
| Battery capacity | 1.5 kWh | 1.5 kWh |
| Energy density | 150 Wh/kg | 200 Wh/kg |
| Radius of operation | 9 km | 9 km |
| Optimal cruise speed | ~80 km/h | ~80 km/h |
| Flight altitude | 150 m | 150 m |
| Engine efficiency | 0.90 | 0.93 |
| Air drag coefficient | 0.5 | 0.3 |

### Trucks (RAM ProMaster / Mercedes Sprinter class)
| Parameter | DV | EV |
|-----------|-----|-----|
| Tare weight | 2.5 ton | 2.5 ton + battery |
| Engine power | 150 kW | 190 kW |
| Battery capacity | – | 80 kWh |
| Engine efficiency | ~45% max | 90% |
| Transmission efficiency | Variable | 90% |
| Charging efficiency | – | 90% |
| Total TTW efficiency | ~25% | ~73% |

---

## Limitations Noted
- UAV energy model not validated with real-world experiments
- Unimodal delivery system (single vehicle type)
- No time windows or delivery time constraints
- Simplified wind model (head/tail wind setting)
- No infrastructure costs or manufacturing emissions
- Does not consider truck-drone hybrid systems

---

## Relevance to Your Work
- **OD Data:** Yes (Berlin road network, realistic customer distributions)
- **Scale:** Single city (Berlin)
- **Comparison vehicle:** Diesel truck and Electric truck (not e-bike)
- **Key differences from your work:**
  - Parcel delivery (2.5 kg) vs food delivery (~1 kg)
  - European context (Germany)
  - Compares against trucks, not e-bikes
  - Uses WTW analysis including upstream electricity

---

## Key Equations

**UAV Power Demand:**
```
P^UAV = P^air + κ·P^lift + P^profile + P^climb + P^int
      = ½·ρ·v³·A·c_air + κ·w·T + ρ·R·v_t³·(1+3·(v/v_t)²)·σ·c_bd/8
        + m·g·v·sin(γ) + P^int
```

**WTW Energy Demand:**
```
Ê^DV = E^DV / ε_wtt^Diesel    (ε_wtt^Diesel ≈ 0.90)
Ê^EV = E^EV / ε_wtt^elec      (ε_wtt^elec ≈ 0.50 in Germany)
Ê^UAV = E^UAV / ε_wtt^elec
```

---

## Key Contribution to Field
- **Most detailed UAV energy model** in literature (accounts for hovering, wind, flight phases)
- First to compare UAVs against **electric trucks** (not just diesel)
- Demonstrates **WTW analysis** importance (TTW alone favors EVs unfairly)
- Shows UAVs have **limited use case**: only rural, low-density, calm weather
- Highlights **hovering as critical energy sink** for delivery drones
- Provides framework for **simulation-based comparison** of delivery systems
