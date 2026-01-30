# Baldisseri et al. (2022) - Truck-based drone delivery system assessment

**Full Title:** Truck-based drone delivery system: An economic and environmental assessment

**Authors:** Andrea Baldisseri, Chiara Siragusa, Arianna Seghezzi, Riccardo Mangiaracina, Angela Tumino

**Journal:** Transportation Research Part D: Transport and Environment, Volume 107, 103296

**DOI:** 10.1016/j.trd.2022.103296

---

## Research Focus
- **Comparison:** Truck-Drone joint delivery vs Diesel Van (DV) vs Electric Van (EV)
- **Scope:** Italy/Europe (suburban/rural areas, 1-20 deliveries/km²)
- **Method:** Life Cycle Assessment (LCA) + Total Cost of Ownership (TCO)
- **Verdict:** Truck-drone system has **lowest emissions in all scenarios**; economically viable **only with drone automation**

---

## Key Methodology

### Delivery System Model
- **Mobile depot concept:** Electric truck carries and launches one drone
- **Routing model:** Based on Campbell et al. (2018) continuous approximation
- **Configuration:** Truck and drone deliver in parallel
- **Drone utilization rate (n):** 0.5 or 1.0 (drone deliveries / truck deliveries)

### LCA System Boundary
- **Cradle-to-grave:** Production → Use → Disposal/Recycling
- **Functional unit:** Single delivered package (2.5 kg)
- **Impact category:** Global Warming Potential (kg CO2e)
- **Time span:** 10 years (truck useful life)

### Energy Models
- **Drone energy:** Adapted from Kirschstein (2020) with four flight phases (takeoff, level flight, hovering, landing)
- **Truck energy:** Physics-based model including air drag, rolling resistance, grade resistance, and inertia
- **Wind:** Four equiprobable directions (head, tail, cross) averaged

### TCO Methodology
- **CAPEX:** Vehicle purchase, drone purchase, battery replacement
- **OPEX:** Fuel/electricity, maintenance, insurance, labor, airspace fees
- **Discount rate:** 7% WACC
- **Three scenarios:** Conservative, Optimistic, Future

---

## Key Findings

### LCA Results (g CO2e/delivery)

**Base case (conservative, low wind, low traffic, low density, high carbon grid):**

| Delivery System | Use Phase | Production/Recycling | **Total LCA** |
|-----------------|-----------|---------------------|---------------|
| Diesel Van | 654 | 76 | **730** |
| Electric Van | 454 | 221 | **675** |
| EV + Drone (n=0.5) | 362 | 192 | **555** |
| EV + Drone (n=1) | 364 | 191 | **555** |

**Key finding:** Truck-drone system has **lowest emissions in ALL scenarios tested**, including:
- High wind speed (12 m/s)
- High traffic (n_acc = 3)
- High carbon intensity grid (Germany: 440.8 g/kWh)
- Low delivery density (1 delivery/km²)

### Emission Breakdown

| Component | Diesel Van | Electric Van | EV + Drone (n=1) |
|-----------|------------|--------------|------------------|
| Truck use phase | 90% | 67% | 55% |
| Truck production | 10% | 12% | 56%* |
| Drone use phase | – | – | 9% |
| Drone + battery production | – | – | 35% |

*Production phase emissions become MORE important for truck-drone due to higher delivery throughput

### TCO Results (€/delivery)

**Cost breakdown (conservative scenario, low wind, low traffic, low density):**

| Delivery System | Cost/delivery |
|-----------------|---------------|
| Diesel Van | ~€1.37 |
| Electric Van | ~€1.45 |
| EV + Drone (n=0.5) | ~€1.95 |
| EV + Drone (n=1) | ~€1.77 |

**Key finding:** Labor cost dominates (>65% of total)

| Scenario | Truck-Drone vs Diesel Van |
|----------|---------------------------|
| Conservative (human pilot needed) | **+37% more expensive** |
| Optimistic (partial automation) | **12-26% savings** |
| Future (full automation) | **36% savings** |

### Sensitivity Analysis Results

**Most impactful variables for LCA:**
1. Delivery density (+60% for high traffic, -62% for suburban)
2. Traffic conditions (+83% for congested traffic)
3. Grid carbon intensity (-87% with low-carbon French grid)

**Most impactful variables for TCO:**
1. Automation level (labor cost reduction)
2. Wind speed (affects conservative scenario most)
3. Battery life (affects battery replacement costs)

### Comparison with Fixed-Depot Drone Delivery

| Study | Configuration | Energy/Delivery |
|-------|---------------|-----------------|
| Figliozzi (2017) | Fixed depot, 13 km avg | 562 Wh |
| Stolaroff (2018) | Fixed depot, 4.3 km | 80-805 Wh |
| Kirschstein (2020) | Fixed depot, 9 km | ~1500 Wh |
| **Present study** | **Truck-drone, 0.2 km drone** | **429 Wh** |

**Critical insight:** Truck-drone system is more efficient because drones only fly very short distances (<1 km), while truck handles long-distance transport.

---

## Key Parameters Used

### Drone Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| Type | Octocopter | Interview |
| Tare weight (without battery) | 8 kg | Interview |
| Battery capacity | 2 × 0.8 kWh | Interview |
| Battery energy density | 0.26 kWh/kg | Interview |
| Payload | 2.5 kg | Kirschstein (2020) |
| Level flight speed | 17 m/s (61 km/h) | Interview |
| Takeoff/landing speed | 2.3 m/s (8.3 km/h) | Interview |
| Flight altitude | 60 m | Interview |
| Hovering time | 30 s | Xu (2017) |
| Useful life | 5 years | Interview |

### Truck Specifications

| Parameter | Diesel Van | Electric Van |
|-----------|------------|--------------|
| Model | Mercedes-Benz Sprinter | WorkHorse C1000 |
| Tare weight | 2,670 kg | 2,940 kg |
| Battery capacity | – | 105 kWh |
| Engine efficiency | Variable | 90% |
| Useful life | 10 years | 10 years |

### Operating Parameters

| Parameter | Value |
|-----------|-------|
| Working days/year | 300 |
| Shift duration | 7.5 h |
| Delivery stop time (truck) | 180 s |
| Delivery densities | 1, 20 deliveries/km² |
| Wind speeds | 0, 12 m/s |
| Acceleration coefficient | 1.5, 3.0 |

### Grid Carbon Intensity

| Country | g CO2e/kWh |
|---------|------------|
| France (low carbon) | 58.5 |
| Germany (high carbon) | 440.8 |

---

## Limitations Noted
- **Single delivery per destination** - limits applicability to urban multi-story buildings
- **Constant truck speed** - simplified trajectory model
- **Straight drone path** - may not reflect real-world flight restrictions
- **No weather downtime** - assumes 300 operating days regardless of conditions
- **Single impact category** - only GWP considered
- **Interview-based drone data** - limited validation

---

## Relevance to Your Work
- **OD Data:** No (uses synthetic delivery area with random customer distribution)
- **Scale:** Generic (suburban/rural Italy)
- **Comparison vehicle:** Diesel van and electric van (not e-bike)
- **Key differences from your work:**
  - **Truck-drone hybrid** vs standalone drone delivery
  - European context (Italy) vs China
  - Parcel delivery (2.5 kg) vs food delivery (~1 kg)
  - Suburban/rural focus vs urban
  - Includes **detailed TCO analysis** with automation scenarios

---

## Key Equations

**Drone energy consumption:**
```
E_UAV = [t_tol × (P_UAV(takeoff) + P_UAV(landing)) + t_lf × P_UAV(level) + (t_hover + t_w) × P_UAV(hover)]
        / (ε_charging × ε_transmission × ε_engine)
      + (2t_tol + t_lf + t_hover + t_w) × P_int / ε_charging
```

**Drone power demand:**
```
P_UAV = P_air + k × P_lift + P_profile + P_climb + P_int
```

**Production phase emissions:**
```
CO2e_truck = e_truck × m_body_truck
CO2e_drone = e_drone × m_body_drone
CO2e_battery = e_battery × cap_battery × n_batteries
```

**Battery emissions allocation:**
```
Emissions per kWh used = GHG_battery / (n_cycles × cap_battery)
```

---

## Key Contribution to Field
- **First comprehensive LCA + TCO** of truck-drone joint delivery system
- Demonstrates **truck as mobile depot** concept is more efficient than fixed-depot drone delivery
- Shows **production phase emissions** are significant for drone systems (35% of total)
- Quantifies **automation level** as critical factor for economic viability
- Provides **detailed sensitivity analysis** across multiple variables
- Highlights that **drone flight distance** (not total delivery distance) determines drone efficiency
- Uses **primary data** from Italian drone manufacturer for realistic parameters
