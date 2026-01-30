# Literature Review: Environmental Assessment of Drone Delivery Systems

## Overview

This review synthesizes findings from eight key studies comparing the environmental impacts of drone delivery systems with conventional ground-based delivery vehicles. The studies span 2017-2025 and cover diverse geographic contexts (USA, Germany, Italy, South Korea, China), delivery scenarios (parcel, grocery, food), and comparison vehicles (diesel/electric trucks, vans, motorcycles, tricycles, e-bikes, personal vehicles).

---

## 1. Studies Reviewed

| Study | Region | Drone vs. | Payload | Method | Key Finding |
|-------|--------|-----------|---------|--------|-------------|
| Figliozzi (2017) | USA | Diesel van, Electric van/truck, Tricycle | 5 kg | Analytical LCA | Tricycles outcompete drones when 2+ customers can be grouped |
| Park et al. (2018) | South Korea | Gasoline/Electric motorcycle | Pizza | LCA (TOTAL software) | Drone GWP is 1/6 of motorcycle; rural benefits 13× more than urban |
| Stolaroff et al. (2018) | USA | Diesel/NG/Electric trucks, Vans, Personal vehicles | 0.5-8 kg | Empirical + LCA | Small drones 23-54% lower GHGs; warehouse energy dominates |
| Kirschstein (2020) | Germany | Diesel/Electric trucks | 2.5 kg | WTW energy model | Drones only competitive in rural, low-density, calm weather |
| Yowtak et al. (2020) | USA | ICEV, BEV | 20.4 kg | Cradle-to-grave LCA | BEV outperforms drone even for short distances; UAV needs 70% cost reduction |
| Rodrigues et al. (2022) | USA | Diesel/Electric trucks/vans, E-cargo bike | 0.5 kg | Empirical (188 flights) | Only e-cargo bikes beat drones on GHG/package |
| Baldisseri et al. (2022) | Italy | Diesel van, Electric van | 2.5 kg | LCA + TCO | Truck-drone hybrid lowest emissions; viable only with automation |
| Li et al. (2025) | China | E-bike | 1 kg | LCA + Simulation | Drones greener for 1-2 orders; e-bikes better for 3-4 orders |

---

## 2. Methodological Approaches

### 2.1 Energy Modeling

**Physics-based theoretical models** are common, accounting for:
- Induced power (lift against gravity) – dominant component
- Body/battery/payload drag
- Rotor profile drag
- Climbing power for altitude changes
- Internal electronics

Key power equation (from Rodrigues et al., Kirschstein):
```
P = P_induced + P_drag + P_profile + P_climb + P_internal
```

**Empirically calibrated models** are rarer but more reliable:
- **Stolaroff et al. (2018):** 1,073 flight segments, first major empirical study
- **Rodrigues et al. (2022):** 188 flights, validated against XGBoost ML model (2.1% error)

### 2.2 Life Cycle Assessment Boundaries

| Study | Production | Use Phase | End-of-Life | Warehouse/Infrastructure |
|-------|------------|-----------|-------------|--------------------------|
| Figliozzi (2017) | ✓ | ✓ | ✓ | ✗ |
| Park et al. (2018) | ✗ | ✓ | ✗ | ✗ |
| Stolaroff et al. (2018) | Battery only | ✓ | ✗ | ✓ (critical) |
| Kirschstein (2020) | ✗ | ✓ (WTW) | ✗ | ✗ |
| Yowtak et al. (2020) | ✓ | ✓ | ✗ | ✗ |
| Rodrigues et al. (2022) | ✗ | ✓ | ✗ | ✗ |
| Baldisseri et al. (2022) | ✓ | ✓ | ✓ | ✗ |
| Li et al. (2025) | ✓ | ✓ | ✓ | ✓ (takeaway cabinets) |

**Key insight:** Studies with full LCA boundaries consistently find production phase contributes 35-52% of drone emissions (Figliozzi 2017, Baldisseri 2022).

### 2.3 Well-to-Wheel vs Tank-to-Wheel

Kirschstein (2020) highlights the importance of Well-to-Wheel (WTW) analysis:
- **Diesel WTT efficiency:** ~90% (fuel production)
- **Electricity WTT efficiency:** ~50% in Germany (varies 35-95% with grid mix)

Ignoring upstream emissions understates the impact of electric vehicles (including drones) in carbon-intensive grids.

---

## 3. Energy Consumption Findings

### 3.1 Energy per Distance (MJ/km)

| Vehicle Type | Energy (MJ/km) | Source |
|--------------|----------------|--------|
| Small drone (0.5 kg) | 0.03-0.08 | Stolaroff, Rodrigues |
| Large drone (8 kg) | 0.27 | Stolaroff |
| E-cargo bicycle | 0.10 | Rodrigues |
| Personal EV | ~0.6 | Stolaroff |
| Electric van | 1.65 | Rodrigues |
| Electric truck | 2.4-3.8 | Stolaroff, Rodrigues |
| Diesel van | 4.9 | Rodrigues |
| Personal car | ~2.5 | Stolaroff |
| Diesel truck | 7.3-11.0 | Stolaroff, Rodrigues |

**Key finding:** Small drones are extremely energy-efficient per distance (~1/100th of diesel trucks), but this advantage diminishes with payload.

### 3.2 Energy per Package

The comparison changes dramatically when considering packages per trip:

| Vehicle | MJ/package | Packages/trip | Source |
|---------|------------|---------------|--------|
| E-cargo bicycle | 0.10 | 1 | Rodrigues |
| Small drone | 0.33 | 1 | Rodrigues |
| Electric van | 0.47 | 3.5 | Rodrigues |
| Diesel van | 1.41 | 3.5 | Rodrigues |
| Electric truck | 1.81 | 2.1 | Rodrigues |
| Diesel truck | 5.24 | 2.1 | Rodrigues |

**Key finding:** Multi-drop delivery eliminates most of the drone's energy advantage. Electric vans with 3.5 packages/trip have comparable energy/package to drones.

### 3.3 Payload Sensitivity

Energy consumption scales with mass^1.5 for multicopters:
```
E ∝ m^1.5 / √ρ
```

| Study | Finding |
|-------|---------|
| Figliozzi (2017) | Drones efficient for small payloads (<5 kg) only |
| Stolaroff (2018) | 0.5 kg drone: 32 J/m; 8 kg drone: 266 J/m (8× higher) |
| Yowtak (2020) | 20 kg payload drone uses 2× energy of BEV |
| Li (2025) | Drones lose advantage for 3-4 orders due to payload weight |

---

## 4. Greenhouse Gas Emissions

### 4.1 Emissions per Delivery (g CO2e)

| Delivery System | Emissions | Context | Source |
|-----------------|-----------|---------|--------|
| E-cargo bicycle | 23 | US average | Rodrigues |
| Small drone | 48-94 | US (varies by grid) | Rodrigues |
| Small drone | 70 | US average | Rodrigues |
| Electric van | 99 | US average | Rodrigues |
| Diesel van | 119 | US average | Rodrigues |
| Electric truck | 373 | US average | Rodrigues |
| Diesel truck | 444 | US average | Rodrigues |
| Personal vehicle | 2100 | US average | Stolaroff |

### 4.2 Grid Carbon Intensity Sensitivity

The electricity grid mix is often the dominant factor for drone emissions:

| Region | Grid (g CO2e/kWh) | Drone Emission Impact |
|--------|-------------------|----------------------|
| France | 59 | 87% lower than German grid |
| California (CAMX) | ~500 | Drones clearly favorable |
| US Average | 638-750 | Drones marginally favorable |
| Germany | 441-500 | Drones need low density to compete |
| China | 600 | Threshold case |
| Korea | ~550 | 2.3× higher than US |
| Missouri (SRMW) | 900-1050 | Large drones worse than diesel |

**Key finding:** Drones are environmentally favorable only where grid carbon intensity is below ~500 g CO2e/kWh (Kirschstein, Stolaroff, Rodrigues).

### 4.3 Production Phase Contribution

Often overlooked, production emissions are significant for drones:

| Component | Contribution to Total | Source |
|-----------|----------------------|--------|
| Drone body | 5-10% | Baldisseri, Figliozzi |
| Drone batteries | 25-35% | Baldisseri |
| Total production | 35-52% | Figliozzi, Baldisseri |
| Equivalent range | 52% of lifetime travel | Figliozzi |

Battery allocation method matters: emissions should be spread across cycle life, not just one trip.

---

## 5. Critical Factors Affecting Competitiveness

### 5.1 Delivery Density

| Density | Drone Competitiveness | Source |
|---------|----------------------|--------|
| 1 delivery/km² (rural) | Highly favorable | Baldisseri, Kirschstein |
| 2 km radius (urban) | Unfavorable vs EV | Kirschstein |
| 8 km radius (suburban) | Competitive | Kirschstein |
| High-density urban | Only with takeaway cabinets | Li |

**Breakeven analysis (Figliozzi 2017):**
- Drone vs Diesel Van: 58-362 customers/route
- Drone vs Electric Van: 10-26 customers/route
- **Drone vs Tricycle: 1.5-2.1 customers/route**

### 5.2 Wind Conditions

| Wind Speed | Energy Impact | Source |
|------------|---------------|--------|
| Calm (0 m/s) | Baseline | All |
| Moderate (6-7 m/s) | +50-100% | Kirschstein |
| High (12 m/s) | +200-300% | Kirschstein, Baldisseri |
| >35 km/h | Battery depleted in 4 min hover | Kirschstein |

### 5.3 Hovering vs Cruising

Hovering is extremely energy-intensive:
- **5 min hovering** can exceed 36% of total flight energy (Rodrigues)
- Drone designs optimized for cruise, not hover (Kirschstein)
- **Takeoff/landing:** 36% of trip energy from vertical flight alone (Rodrigues)

### 5.4 Delivery Distance

| Configuration | Avg Drone Flight | Energy/Delivery | Source |
|---------------|------------------|-----------------|--------|
| Fixed depot | 4-13 km | 80-1500 Wh | Various |
| **Truck-drone** | **0.2 km** | **429 Wh** | Baldisseri |

**Key insight:** Truck-drone hybrid is more efficient because drones fly very short distances while trucks handle long-haul.

---

## 6. Urban vs Rural Comparison

Park et al. (2018) provides the only systematic urban-rural comparison:

| Metric | Urban (Seoul) | Rural (Pyeongchang) | Rural/Urban Ratio |
|--------|---------------|--------------------|--------------------|
| Avg delivery distance | 0.8 km | 8.8 km | 11× |
| GWP reduction | 0.024 kg CO2e | 0.32 kg CO2e | **13.4×** |
| PM2.5 reduction | 1.56×10⁻⁶ kg | 2.20×10⁻⁵ kg | **14.1×** |

**Key finding:** Environmental improvement from drone adoption is **13× higher in rural areas** than urban areas.

---

## 7. Economic Considerations

### 7.1 Total Cost of Ownership

| System | Cost/Delivery | Key Cost Driver | Source |
|--------|---------------|-----------------|--------|
| Diesel Van | €1.37 | Fuel (35%) | Baldisseri |
| Electric Van | €1.45 | Purchase (30%) | Baldisseri |
| Drone (manual pilot) | €1.77-1.95 | **Labor (>65%)** | Baldisseri |
| Drone (automated) | €0.88 | Capital (60%) | Baldisseri |

### 7.2 Automation Impact

| Scenario | Drone vs Diesel Van | Source |
|----------|---------------------|--------|
| Human pilot required | +37% more expensive | Baldisseri |
| Partial automation | 12-26% savings | Baldisseri |
| Full automation | **36% savings** | Baldisseri |

### 7.3 Capital vs Operating Costs

| System | Capital Cost | 2-Year Total | Source |
|--------|--------------|--------------|--------|
| ICEV (used) | $0 | $9,250 | Yowtak |
| BEV | $44,500 | $51,420 | Yowtak |
| UAV Fleet (10 units) | $172,500 | **$187,600** | Yowtak |

**Key finding:** UAV costs need to decrease by **70%** to compete with BEV (Yowtak 2020).

---

## 8. Comparison with E-bikes/Bicycles

Only three studies directly compare drones with bicycles/e-bikes:

| Study | Comparison | Finding |
|-------|------------|---------|
| Figliozzi (2017) | Drone vs Electric Tricycle | Tricycle wins when 2+ customers grouped |
| Rodrigues et al. (2022) | Drone vs E-cargo Bicycle | **Only e-cargo bikes beat drones** |
| Li et al. (2025) | Drone vs E-bike | Drones greener for 1-2 orders; e-bikes better for 3-4 |

**E-cargo bicycle advantages:**
- 23 g CO2e/package vs 70 g CO2e for drone (Rodrigues)
- Multi-delivery capability eliminates drone's single-trip efficiency
- Lower capital cost, simpler infrastructure

**Drone advantages over e-bikes:**
- Faster delivery (40 km/h vs 25 km/h)
- No road congestion effects
- Better for long-distance rural delivery
- No rider fatigue or labor issues

---

## 9. Warehouse and Infrastructure Energy

A critical factor often neglected (only Stolaroff 2018 and Li 2025 address this):

| Infrastructure Component | Energy/Package | % of Total | Source |
|--------------------------|----------------|------------|--------|
| Warehouse (electricity + gas) | 0.35 kWh + 0.93 MJ | **>50%** | Stolaroff |
| Takeaway cabinets (24/7) | – | **49-65%** | Li |
| Drone flight | ~0.08 kWh | 30-50% | Various |

**Key finding:** Doubling drone range reduces required warehouse nodes by **3.5×**, significantly cutting infrastructure emissions (Stolaroff).

---

## 10. Research Gaps and Limitations

### 10.1 Common Limitations Across Studies

| Limitation | Studies Affected |
|------------|------------------|
| No weather downtime modeled | Most except Yowtak |
| Simplified wind models | All except empirical studies |
| Single impact category (GWP only) | Most except Park |
| No airspace/regulatory constraints | All |
| No multi-drop drone scenarios | All |
| Limited validation of energy models | Most except Stolaroff, Rodrigues |

### 10.2 Geographic Gaps

| Region | Studies | Gap |
|--------|---------|-----|
| USA | 4 | Well-covered |
| Europe | 2 | Moderate coverage |
| East Asia | 2 | Limited (Korea, China only) |
| Developing countries | 0 | **No coverage** |
| Dense urban Asia | 1 | **Major gap** |

### 10.3 Vehicle Type Gaps

| Vehicle | Studies Comparing |
|---------|-------------------|
| Diesel truck/van | 6 |
| Electric truck/van | 5 |
| Personal vehicle | 2 |
| Motorcycle | 1 |
| Tricycle | 1 |
| **E-bike** | **1** (Li 2025 only) |
| **E-scooter** | **0** |

**Critical gap:** E-bikes dominate Asian food delivery but are underrepresented in literature.

---

## 11. Key Takeaways for Chinese Food Delivery Context

Based on the synthesis:

1. **Grid carbon intensity is critical:** China's ~600 g CO2e/kWh grid puts drones at the threshold of competitiveness. Regional variation within China matters.

2. **Payload matters enormously:** Food delivery (~1 kg) favors drones more than grocery delivery (~5-20 kg). Small drones are ~8× more efficient than large ones.

3. **Delivery density is decisive:** High-density urban areas favor e-bikes with multi-drop capability. Drones may be favorable for suburban/campus delivery where density is lower.

4. **Production emissions matter:** 35-50% of lifecycle emissions from drone production/batteries. Short drone lifespan amplifies this effect.

5. **Infrastructure emissions overlooked:** Charging stations, distribution centers, and takeaway cabinets can dominate total emissions if not optimized.

6. **E-bike is the key comparator:** Only one study (Li 2025) compares drones to e-bikes directly. More research needed for the dominant mode in Chinese food delivery.

7. **Distance thresholds:** Drones become more favorable beyond ~4-5 km delivery distances where e-bike efficiency drops and delivery time advantage grows.

8. **Economic viability requires automation:** Current labor costs make drone delivery 37% more expensive than traditional methods. Full automation could achieve 36% savings.

---

## 12. References

1. Baldisseri, A., Siragusa, C., Seghezzi, A., Mangiaracina, R., & Tumino, A. (2022). Truck-based drone delivery system: An economic and environmental assessment. *Transportation Research Part D*, 107, 103296.

2. Figliozzi, M. A. (2017). Lifecycle modeling and assessment of unmanned aerial vehicles (Drones) CO2e emissions. *Transportation Research Part D*, 57, 251-261.

3. Kirschstein, T. (2020). Comparison of energy demands of drone-based and ground-based parcel delivery services. *Transportation Research Part D*, 78, 102209.

4. Li, Z., Zhou, S., Wang, B., Zhang, T., & Guo, S. (2025). Beyond the last-mile: Environmental and economic assessment of the upcoming drone takeaway delivery system. *Sustainable Cities and Society*, 120, 106134.

5. Park, J., Kim, S., & Suh, K. (2018). A comparative analysis of the environmental benefits of drone-based delivery services in urban and rural areas. *Sustainability*, 10(3), 888.

6. Rodrigues, T. A., Patrikar, J., Oliveira, N. L., Matthews, H. S., Scherer, S., & Samaras, C. (2022). Drone flight data reveal energy and greenhouse gas emissions savings for very small package delivery. *Patterns*, 3, 100569.

7. Stolaroff, J. K., Samaras, C., O'Neill, E. R., Lubers, A., Mitchell, A. S., & Ceperley, D. (2018). Energy use and life cycle greenhouse gas emissions of drones for commercial package delivery. *Nature Communications*, 9, 409.

8. Yowtak, K., Imiola, J., Andrews, M., Cardillo, K., & Skerlos, S. (2020). Comparative life cycle assessment of unmanned aerial vehicles, internal combustion engine vehicles and battery electric vehicles for grocery delivery. *Procedia CIRP*, 90, 244-250.
