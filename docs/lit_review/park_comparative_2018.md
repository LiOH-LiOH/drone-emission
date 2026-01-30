# Park et al. (2018) - Environmental Benefits of Drone-Based Delivery in Urban and Rural Areas

**Full Title:** A Comparative Analysis of the Environmental Benefits of Drone-Based Delivery Services in Urban and Rural Areas

**Authors:** Jiyoon Park, Solhee Kim, Kyo Suh

**Journal:** Sustainability, Volume 10, 888

**DOI:** 10.3390/su10030888

---

## Research Focus
- **Comparison:** Drone vs Gasoline Motorcycle vs Electric Motorcycle
- **Scope:** South Korea (Seoul urban vs Pyeongchang rural)
- **Method:** Life Cycle Assessment (LCA) using TOTAL software
- **Verdict:** Yes - Drone GWP is 1/6 of motorcycle; rural areas benefit 13× more than urban

---

## Key Methodology

### Delivery Scenario
- **Item:** Pizza delivery (uniform weight, light packaging)
- **Urban area:** Yangcheon-gu, Seoul (pop. density: 26,464/km²)
- **Rural area:** Pyeongchang-gun, Gangwon-do (pop. density: 27.6/km²)
- **Distance estimation:** Elementary school locations as residential proxies

### Vehicle Specifications

| Parameter | Drone (MD4-1000) | Motorcycle (Honda Super Cub) | Electric MC (Valencia) |
|-----------|------------------|------------------------------|------------------------|
| Energy source | Electricity | Gasoline | Electricity |
| Battery | 22.2V, 13,000 mAh | – | 72V, 30,000 mAh |
| Flight/Speed | 12 m/s, 45 min | 63.5 km/L | 60 km range |
| Manufacturer | Microdrones | Honda | Green Mobility |

### LCA System Boundary
- Electricity/gasoline production (Well-to-Tank)
- Gasoline combustion emissions (Tank-to-Wheel)
- **Software:** TOTAL (Korean government LCA tool)
- **Databases:** Korea National LCI DB, US LCI DB (NREL)

### Impact Categories
10 categories including GWP, AP, EP, FAETP, HTP, MAETP, ODP, POCP, TETP, plus PM2.5

---

## Key Findings

### Per-Distance Emissions (1 km delivery)

| Vehicle | GWP (kg CO2-eq) | PM2.5 (kg PM2.5-eq) |
|---------|-----------------|---------------------|
| Drone | **4.41 × 10⁻³** | **9.65 × 10⁻⁷** |
| Gasoline Motorcycle | 2.85 × 10⁻² | 2.09 × 10⁻⁶ |
| Electric Motorcycle | 1.78 × 10⁻² | **3.90 × 10⁻⁶** |

**Key insights:**
- Drone GWP is **1/6 (16.5%)** of gasoline motorcycle
- Drone PM2.5 is **~half** of gasoline motorcycle
- **Electric motorcycles produce MORE PM2.5** than gasoline motorcycles (due to electricity production)

### Emission Breakdown
- **Gasoline combustion** produces 24× more GHG than gasoline production
- **Electricity production** generates more PM2.5 than gasoline production + combustion combined

### Urban vs Rural Comparison

| Region | Avg Distance (drone) | GWP Reduction | PM2.5 Reduction |
|--------|---------------------|---------------|-----------------|
| Yangcheon-gu (urban) | 0.80 km round-trip | 2.41 × 10⁻² kg CO2-eq | 1.56 × 10⁻⁶ kg PM2.5-eq |
| Pyeongchang-gun (rural) | 8.83 km round-trip | 0.323 kg CO2-eq | 2.20 × 10⁻⁵ kg PM2.5-eq |
| **Rural/Urban ratio** | **11×** | **13.4×** | **14.1×** |

**Critical finding:** Environmental improvement from drone adoption is **13× higher in rural areas** than urban areas.

### National Grid Comparison (Korea vs US)

| Metric | Korea | US | Ratio |
|--------|-------|-----|-------|
| GWP per km drone | 4.41 × 10⁻³ kg CO2-eq | 1.92 × 10⁻³ kg CO2-eq | **2.3×** |
| PM2.5 per km drone | 9.65 × 10⁻⁷ kg PM2.5-eq | 7.77 × 10⁻⁷ kg PM2.5-eq | **1.24×** |

**Key insight:** Korea's electricity grid produces 2.3× more GWP than US grid.

### Renewable Energy Sensitivity
With 13.4% renewable energy (Korea 2035 target):
- Photovoltaic: **12% GWP reduction** (3.88 × 10⁻³ kg CO2-eq/km)
- Wind: **13% GWP reduction** (3.84 × 10⁻³ kg CO2-eq/km)
- Solar thermal: **13% GWP reduction** (3.85 × 10⁻³ kg CO2-eq/km)

### Weighted Environmental Impact
Considering all 10 impact categories:
- Drone: **4.37 × 10⁻⁷** (weighted)
- Motorcycle: **1.46 × 10⁻⁶** (weighted)
- **Drone is 1/12 of motorcycle's total environmental impact**

---

## Limitations Noted
- Based on product specifications, not empirical flight data
- Food delivery focus only (pizza)
- Does not consider drone limitations (weather, regulations, airspace)
- LCI database baseline years are old (Korea: 2000, US: 2008)
- Drone can only deliver one item per trip (no batching)
- Does not include vehicle manufacturing/disposal phases

---

## Relevance to Your Work
- **OD Data:** Yes (uses elementary school locations as residential proxies)
- **Scale:** Two regions (urban Seoul vs rural Gangwon-do)
- **Comparison vehicle:** Motorcycle (similar to e-bike in use case)
- **Key differences from your work:**
  - Compares drone vs gasoline/electric motorcycle (not e-bike)
  - Korea context (similar grid issues to China)
  - Food delivery focus (same as your study)
  - Rural vs urban analysis (your work focuses on single city)
  - Does not use real OD data from delivery platforms

---

## Key Parameters Used

| Parameter | Value |
|-----------|-------|
| Drone model | Microdrones MD4-1000 |
| Battery capacity | 22.2V, 13,000 mAh (288.6 Wh) |
| Flight time | 45 min |
| Flight speed | 12 m/s (43.2 km/h) |
| Urban delivery distance | 0.40 km (one-way, straight line) |
| Rural delivery distance | 4.42 km (one-way, straight line) |
| Bypass coefficient Seoul | 1.21 |
| Bypass coefficient Gangwon | 1.43 |
| Korea grid emissions | ~0.55 kg CO2e/kWh (implied) |
| US grid emissions | ~0.24 kg CO2e/kWh (implied) |

---

## Key Equations

**Drone energy consumption:**
```
E_drone = Battery capacity × (distance / (speed × flight_time))
```

**Environmental impact per delivery:**
```
Impact = (LCI_electricity × energy_consumption) × distance
```

**Bypass coefficient application:**
```
Distance_motorcycle = Distance_straight × bypass_coefficient
```

---

## Key Contribution to Field
- **First urban vs rural comparison** of drone delivery environmental benefits
- Shows **rural areas benefit 13× more** from drone adoption than urban areas
- **Includes electric motorcycle** comparison (often missing in literature)
- Highlights **PM2.5 trade-off** for electric vehicles (may be worse than gasoline)
- Provides **national grid sensitivity** analysis for drone emissions
- Demonstrates importance of **renewable energy expansion** for electric delivery vehicles
