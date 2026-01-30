# Yowtak et al. (2020) - Comparative LCA of UAVs, ICEVs and BEVs for Grocery Delivery

**Full Title:** Comparative life cycle assessment of unmanned aerial vehicles, internal combustion engine vehicles and battery electric vehicles for grocery delivery

**Authors:** Kris Yowtak, Justin Imiola, Michael Andrews, Keith Cardillo, Steven Skerlos

**Journal:** Procedia CIRP, Volume 90, pp. 244-250 (27th CIRP LCE Conference)

**DOI:** 10.1016/j.procir.2020.02.003

---

## Research Focus
- **Comparison:** Drone vs ICEV (gasoline van) vs BEV (Tesla Model 3)
- **Scope:** Generic (Ann Arbor, Michigan, USA)
- **Method:** Life Cycle Assessment (cradle-to-grave for UAV, use-phase comparison)
- **Verdict:** Mixed - UAV+ICEV better than ICEV-only for GWP, but BEV outperforms both

---

## Key Methodology

### Delivery Service Model
- **Primary scenario:** UAV operates during good weather (179 days/year), ICEV supplements during bad weather (186 days/year)
- **Reference 1:** ICEV-only (365 days/year)
- **Reference 2:** BEV-only (365 days/year)
- Delivery radius: 3.21 km from Whole Foods Market
- Payload: 20.4 kg grocery box (Amazon Prime Pantry)

### Functional Unit
- Number of grocery deliveries over 2-year UAV lifetime
- 10 UAV fleet making 17,900 deliveries
- Total distance: 114,918 km over 2 years
- Ground vehicles travel 1.34× UAV distance due to road network

### LCA Tools
- Umberto software with Ecoinvent database
- GREET model for ICEV emissions
- TRACI 2.1 impact methodology (US EPA)

### Impact Categories
- Global Warming Potential (GWP, CO2-eq)
- Photochemical Oxidant Formation Potential (POFP, NOx-eq)
- Respiratory Effects (RESP, PM2.5-eq)

---

## Key Findings

### UAV Life Cycle Phases
Use phase dominates for most impact categories:
- **GWP:** ~95% from use phase
- **POFP:** ~90% from use phase
- **RESP:** ~85% from use phase

### Use Phase Comparison (Ann Arbor, MI)

| Impact Category | UAV + ICEV | ICEV-Only | BEV-Only |
|-----------------|------------|-----------|----------|
| GWP (1000 kg CO2-eq) | ~95 | ~105 | ~55 |
| POFP (kg NOx-eq) | ~18 | ~12 | ~8 |
| RESP (kg PM2.5-eq) | ~22 | ~8 | ~18 |

**Key findings:**
- UAV+ICEV has **10% lower GWP** than ICEV-only
- UAV+ICEV has **73% higher GWP** than BEV-only
- UAV+ICEV has **higher POFP and RESP** than both alternatives
- ICEV-only has lowest respiratory effects (due to vehicle emission controls)

### Energy Consumption

| System | 2-Year Energy (MJ) | Relative |
|--------|-------------------|----------|
| UAV + ICEV | 154,656 | 0.91× |
| BEV-only | 169,506 | 1.00× |
| UAV-only (hypothetical) | 315,360 | **1.86×** |

**Critical finding:** Even at 100% utilization, UAV fleet consumes nearly **2× more energy** than BEV.

### Weather Limitations

| Location | Days Below 0°C | Rain Days | UAV Flight Days |
|----------|---------------|-----------|-----------------|
| Ann Arbor, MI | 129 | 114 | **179** |
| Los Angeles, CA | 0 | 33 | **349** |
| Grand Forks, ND | 184 | 62 | **150** |

### Sensitivity Analysis
- Grid mix and weather conditions change absolute values but not relative rankings
- Even with "clean" grid (Los Angeles), BEV still outperforms UAV
- UAV would need **13.5 km range** (vs current ~6 km) to match BEV in LA scenario
- This would require **doubling flight speed** (24→50 km/h) or **doubling flight time** (20→42 min)

### Economic Analysis

| Cost Component | ICEV | BEV | UAV Fleet |
|----------------|------|-----|-----------|
| Capital | $0 (pre-owned) | $44,500 | $172,500 |
| Fuel/Electricity | $9,250 | $6,920 | $6,850 |
| **Total (2 years)** | $9,250 | $51,420 | **$187,600** |

**Key finding:** UAV costs would need to decrease by **70%** to be competitive with BEV.

---

## Limitations Noted
- Single UAV model (Harris Aerial HX8 Power XL)
- Cold weather effects on ICEV/BEV efficiency not considered
- No infrastructure costs included
- Limited to grocery delivery scenario
- 2-year UAV lifetime assumption may be optimistic

---

## Relevance to Your Work
- **OD Data:** No (fixed delivery radius from single store)
- **Scale:** Generic (single delivery point)
- **Comparison vehicle:** ICEV (gasoline) and BEV (Tesla Model 3), not e-bike
- **Key difference from your work:**
  - High payload (20.4 kg) vs typical food delivery (~1 kg)
  - USA electricity grid (cleaner than China)
  - BEV comparison (Tesla Model 3) rather than e-bike
  - Weather-constrained operation model

---

## Key Parameters Used

| Parameter | UAV (Harris HX8) | ICEV | BEV (Tesla M3) |
|-----------|------------------|------|----------------|
| Max Payload | 35 kg | – | – |
| Battery Capacity | 120,000 mAh | – | – |
| Energy Consumption | 1,342 kJ/km | 11.22 km/L | ~1,475 kJ/km |
| Speed | 24.1 km/h | – | – |
| Flight/Drive Time | 15 min round-trip | – | – |
| Range | ~6.4 km | 695 km | 499 km |
| Capital Cost | $15,000/unit | Pre-owned | $44,500 |
| Service Life | 2 years | – | – |

---

## Key Contribution to Field
- First study comparing high-payload UAV against **BEV** (not just ICEV)
- Demonstrates that **BEV outperforms UAV** even for short-distance delivery
- Highlights **weather limitations** as critical factor for UAV operations
- Shows UAV **energy inefficiency** for heavy payloads
- Provides **economic analysis** showing UAV not cost-competitive
- Concludes high-payload UAVs need significant improvements in range, efficiency, and cost
