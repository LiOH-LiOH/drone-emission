# Li et al. (2025) - Beyond the last-mile

**Full Title:** Beyond the last-mile: Environmental and economic assessment of the upcoming drone takeaway delivery system

**Authors:** Zhi Li, Siqi Zhou, Bohan Wang, Tingxi Zhang, Shuang Guo

**Journal:** Sustainable Cities and Society, Volume 120, 106134

**DOI:** 10.1016/j.scs.2025.106134

---

## Research Focus
- **Comparison:** Drone vs E-bike for takeaway (food) delivery
- **Scope:** Single city (Qixia District, Nanjing, China, 12×12 km)
- **Method:** Life Cycle Assessment (LCA) + Multi-Agent Based Simulation (MABS)
- **Verdict:** Mixed - drones greener for 1-2 orders/flight, e-bikes better for 3-4 orders

---

## Key Methodology

### Simulation Model
- Multi-Agent Based Simulation (MABS) using AnyLogic
- 5 agent types: Distribution Center, Takeaway Cabinet, E-bike, Drone, Consumer
- 100 e-bikes, 100 drones, 10 distribution centers, 120 takeaway cabinets
- Generated over 1 million data points

### LCA System Boundary
- Cradle-to-grave: production, use, and disposal phases
- Functional unit: delivery of a single takeaway order (assumed 1kg average)
- GHG emissions reported in g CO2e (100-year GWP)

### Energy Models
- **E-bike:** 60V20A lithium iron phosphate battery, ~25 km/h speed
- **Drone:** MD4-3000 model, lithium battery, ~40 km/h speed
- Grid carbon intensity: 0.6 kg CO2e/kWh

---

## Key Findings

### GHG Emissions (Use Phase)
- Drones: 38-116 g CO2e/order
- E-bikes: 31-75 g CO2e/order
- **Drones emit 25%-54% MORE than e-bikes in use phase**

### GHG Emissions (Life Cycle)
- Drones: 159-364 g CO2e/order (1-2 orders/flight)
- E-bikes: 182-375 g CO2e/order
- **Drones emit 1%-12% LESS when delivering 1-2 orders**
- For 3-4 orders: drones emit MORE due to payload weight sensitivity

### Takeaway Cabinets
- Major contributor: 49-65% of total emissions
- Due to 24/7 continuous operation

### Economic Analysis
- Drone delivery: 2.5-3.6× higher revenue than e-bike
- E-bike labor costs: 80-89% of total costs
- Drone procurement: 39-40% of total costs
- Drone system achieves positive returns within 2 years

### Optimal Scenarios
- **Collaborative delivery (S4):** Best at 4-7 km zonal division
  - 4.59% GHG reduction + 12.11% economic benefit increase at 5km partition
- **Solar-powered drones:** 60-70% GHG reduction per order

---

## Limitations Noted
- Simplified assumptions about weather and road congestion
- Did not explore social impacts (job displacement, equity, public acceptance)
- Results may vary with different road network densities

---

## Relevance to Your Work
- **OD Data:** Yes (real scene in Nanjing with actual mall locations)
- **Scale:** Single city (12×12 km area)
- **Comparison vehicle:** E-bike (same as your study)
- **Key difference from your work:** They use takeaway cabinets as intermediaries; your work compares direct delivery

---

## Key Parameters Used
| Parameter | E-bike | Drone |
|-----------|--------|-------|
| Speed | ~25 km/h | ~40 km/h |
| Battery | 60V20A LiFePO4 | Lithium |
| Price | 4,000 CNY | 60,000 CNY |
| Service life | 4 years | 4 years |
| Delivery time | 40-100 min | 20-40 min |
