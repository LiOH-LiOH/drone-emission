# Rodrigues et al. (2022) - Drone flight data reveal energy and GHG savings

**Full Title:** Drone flight data reveal energy and greenhouse gas emissions savings for very small package delivery

**Authors:** Thiago A. Rodrigues, Jay Patrikar, Natalia L. Oliveira, H. Scott Matthews, Sebastian Scherer, Constantine Samaras

**Journal:** Patterns, Volume 3, 100569

**DOI:** 10.1016/j.patter.2022.100569

---

## Research Focus
- **Comparison:** Drone vs Diesel Truck / Electric Truck / Diesel Van / Electric Van / E-Cargo Bicycle
- **Scope:** Generic (USA, with regional grid variation)
- **Method:** Empirical flight data (188 flights) + First-principles model + Machine Learning
- **Verdict:** Mixed - Drones up to 94% lower energy/package than trucks; only e-cargo bikes have lower GHGs/package

---

## Key Methodology

### Empirical Data Collection
- **188 actual flights** with DJI Matrice 100 quadcopter
- Altitudes: 25, 50, 75, 100 m
- Speeds: 4, 6, 8, 10, 12 m/s
- Payloads: 0, 250, 500 g
- Wind conditions: 2-16 knots
- Sensors: anemometer, current/voltage monitor, GPS, accelerometer
- Data frequency: ~5 Hz

### Energy Model
Based on induced power (hover, no-wind condition):

```
E = (b₁ · m^1.5 / √ρ + b₀) · t
```

Where:
- m = total mass (drone + payload)
- ρ = air density
- t = flight duration
- b₁, b₀ = regression coefficients

**Model coefficients (± bootstrap SE):**

| Flight Regime | b₁ | b₀ | R² |
|---------------|-----|-----|-----|
| Takeoff | 80.4 ± 2.6 | 13.8 ± 18.9 | 0.84 |
| Cruise | 68.9 ± 2.0 | 16.8 ± 15.0 | 0.85 |
| Landing | 71.5 ± 1.7 | -24.3 ± 12.5 | 0.90 |

### Model Validation
- Compared against XGBoost machine learning model
- Absolute relative error: **2.1%** on test set (68 flights)
- Open model and data available on Figshare/Zenodo

---

## Key Findings

### Drone Energy Consumption
- **~0.08 MJ/km** (0.5 kg payload, 12 m/s cruise, 100m altitude)
- **~0.33 MJ/package** (2 km one-way delivery, 4 km round trip)
- **~70 g CO2e/package** (US average non-baseload grid)

### Key Insight: Speed Has Little Effect on Power
Within tested range (4-12 m/s), average power consumption varies only ~5%:
- Power is dominated by induced power (overcoming gravity)
- But faster speed = shorter flight time = less total energy per delivery
- **Faster is more efficient** for quadcopters (within operating limits)

### Takeoff/Landing Energy Penalty
- Vertical flight (takeoff + landing) accounts for **36% of total energy**
- Could be reduced by **95%** with 5m takeoff (e.g., from building rooftop)
- This would reduce total trip energy by **34%** or increase range by 50%

### Regional Grid Emissions (US)

| Region | Grid CO2e (g/kWh) | Drone g CO2e/package |
|--------|-------------------|----------------------|
| New York (cleanest) | 399 | **48.5** |
| US Average | 638 | **70.1** |
| Central Midwest (dirtiest) | 900 | **93.8** |

**Up to 93% higher emissions** in carbon-intensive regions.

### Comparison with Other Vehicles

**Energy Consumption (MJ/km):**

| Vehicle | MJ/km |
|---------|-------|
| Medium-duty diesel truck | 11.00 |
| Small diesel van | 4.90 |
| Medium-duty electric truck | 3.80 |
| Small electric van | 1.65 |
| Electric cargo bicycle | **0.10** |
| Small quadcopter drone | **0.08** |

**Energy per Package (MJ/package):**

| Vehicle | MJ/package |
|---------|------------|
| Medium-duty diesel truck | 5.24 |
| Medium-duty electric truck | 1.81 |
| Small diesel van | 1.41 |
| Small electric van | 0.47 |
| **Small quadcopter drone** | **0.33** |
| Electric cargo bicycle | **0.10** |

**GHG Emissions per Package (g CO2e/package):**

| Vehicle | g CO2e/package |
|---------|----------------|
| Medium-duty diesel truck | 443.6 |
| Medium-duty electric truck | 372.6 |
| Small diesel van | 119.2 |
| Small electric van | 98.7 |
| **Small quadcopter drone** | **70.1** |
| **Electric cargo bicycle** | **23.4** |

### Emission Reduction Potential
Drones can reduce compared to:
- Diesel trucks: **94% energy**, **84% GHG** per package
- Electric vans: **31% energy**, **29% GHG** per package
- **Only e-cargo bicycles beat drones** on GHG/package

### Delivery Density Breakeven Analysis

| Vehicle | Packages/km needed to match drone |
|---------|-----------------------------------|
| Medium-duty diesel truck | 33.8 (16× base) |
| Small diesel van | 15.0 (4.3× base) |
| Medium-duty electric truck | 11.7 (5.6× base) |
| Small electric van | 5.1 (1.5× base) |
| Electric cargo bicycle | 0.3 (0.3× base) |
| Drone (base case) | 0.25 |

---

## Limitations Noted
- Very small drone (0.5 kg max payload) - not representative of heavier delivery drones
- Drone not optimized for energy efficiency
- Airspace regulations (no-fly zones, avoiding people/vehicles) not considered
- System-wide effects (warehousing, infrastructure) not included
- Wind conditions limited to 2-16 knots

---

## Relevance to Your Work
- **OD Data:** No (fixed delivery scenarios)
- **Scale:** Generic (applicable to any location with grid adjustment)
- **Comparison vehicle:** Includes **e-cargo bicycle** (similar to e-bike!)
- **Key differences from your work:**
  - Very small drone (0.5 kg) vs your assumed ~1 kg food delivery
  - Includes e-cargo bicycle (closest comparison to your e-bike)
  - USA context, but provides regional grid sensitivity
  - **Empirical data** rather than theoretical model

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Drone model | DJI Matrice 100 |
| Drone mass (empty + sensors) | ~2.4 kg |
| Max payload tested | 0.5 kg |
| Battery capacity | 130 Wh (LiPO TB48D) |
| Cruise altitude | 100 m |
| Cruise speed | 4-12 m/s |
| Takeoff speed | 2.5 m/s |
| Landing speed | 2.0 m/s |
| One-way delivery range | ~2 km |
| Round-trip energy | ~74.5 Wh |
| Charging efficiency | 88% |
| Transmission losses | 6.5% |

---

## Key Equations

**Induced Power (hover, no-wind):**
```
Pᵢ = (mg)^1.5 / √(2ρA)
```

**Total Energy:**
```
E = Σ (b₁ · m^1.5/√ρ + b₀) · t
    for regimes: takeoff, cruise, landing
    for loads: loaded, unloaded
```

**Two-way Range:**
```
d = [Emax - (E_takeoff^l + E_takeoff^u + E_landing^l + E_landing^u)] · Vcr
    / [b₁(m^1.5/√ρ)^l + (m^1.5/√ρ)^u) + 2b₀]
```

---

## Key Contribution to Field
- **First large-scale empirical study** (188 flights) of drone energy consumption
- **Open data and model** available for other researchers
- **Includes e-cargo bicycle** comparison (often missing in literature)
- Shows **regional grid sensitivity** is critical for GHG assessment
- Demonstrates **speed has minimal effect** on power within operating range
- Identifies **vertical flight (takeoff/landing) as major energy sink** (36%)
- Provides **practical coefficients** for estimating drone energy use
