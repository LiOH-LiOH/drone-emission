# Li 2025 运输运营成本与三目标替代分析

2026-09-29：本页保留为运输运营成本对照模型。新增的全成本扩展先按南京全城需求配置车队和设施，见[南京全成本试验](full_cost_nanjing_pilot.md)；未覆盖或重跑原313城结果。

## 已确认边界

统一来源：Li et al. (2025), *Beyond the last-mile: Environmental and economic assessment of the upcoming drone takeaway delivery system*, Sustainable Cities and Society 120, 106134，DOI: [10.1016/j.scs.2025.106134](https://doi.org/10.1016/j.scs.2025.106134)。

用户确认采用论文模型假设，不再逐项追溯独立实测出处。成本仅覆盖运输环节：骑手报酬、充电电费和电池更换。不计车辆或设施初购折旧、商场集餐、柜体运行、独立远程监管和维修。原文商场集餐是两种方式的共同环节；本项目不新增该服务链。

分别逐 OD 求排放、成本、运输时间最优策略，目标相等时保留电动车。没有容量或排队约束。运输时间沿用现有单程字段（无人机含升降、悬停），不含返程、等待、取餐或交接。费用采用完整往返任务能耗，不能与单程时间指标混淆。

## 来源与数值

用户提供原件：`docs/ref/sup_info/li_2025_supplement.docx`；文本提取：同目录 `li_2025_supplement.md`。原始下载文件保留。

原件 SHA256：`6eb40bae3f7e230e2b1c4218a1faf0f4f0f45e31a220f75c0341599595998ad0`。

| 参数 | 采用值 | 来源／适配 |
|---|---:|---|
| 电价 | 0.5 CNY/kWh | S4、S7 |
| 购电换算效率 | 0.81 | 正文式(10)、(11)，电池侧能量除以此值 |
| 骑手报酬比例 | 配送费 × 0.7 | 正文式(8) |
| 配送费 | 3 km内2.8元；超过部分1.4元/km | 正文式(9) |
| 电动车／无人机电池价格 | 400／1000元 | S7 |
| 电池寿命 | 2年 | S7；S11明确年限 |
| 残值比例 | 5% | 正文式(7)、S8 |
| 电动车／无人机经济电池库存 | 每车2／3套 | 本项目映射：S7的200／300套除以各100辆／架 |

S7库存与S4物理电池数量5／4属于不同口径，不相乘。采用项目现有速度30／60／20 km/h等运输参数，不替换成来源研究速度。S11的2→4年情景本轮不扫描。

## 每单计算

令道路单程距离为 r km，空中单程距离为 a km：

```text
rider(r) = 0.7 × [2.8 + 1.4 × max(r − 3, 0)]
electricity_m = round_trip_energy_m / 0.81 × 0.5
battery_m = 0.95 × battery_price_m × stock_per_vehicle_m
            / battery_life_years / annual_orders_per_vehicle_m
cost_ebike = rider + electricity_ebike + battery_ebike
cost_drone = electricity_drone + battery_drone
```

骑手费用距离不乘往返系数或合单系数。电动车能耗和资源分摊继续使用 OD 中的 `ebike_kappa`。直接读取能耗计算结果，电网因子为零也可算成本，不用排放量除以电网因子反推。

每车年服务量沿用既有生命周期模型：`每日工作小时 × 60 × 年工作日 / 往返任务分钟`。任务分钟为 `2 × 有效单程距离 / 速度 × 60 + 装卸分钟`；这延续了既有无人机资源分摊不另加升降与悬停的简化。工作小时、工作日、装卸时间取现有车辆参数。此处装卸仅用于资源分摊，不进入运输时间绩效指标。

## 代码与结果

配置：`experiment.toml` 的 `[transport_cost]`。

入口：`python -m drone_emission --env remote analyze --only cost-time-substitution`。输入改变时按现有分析工作流加 `--force`，只运行所选模块。

输出在工作目录 `analysis/cost_time_substitution/`：

- `city_strategies.csv`：锁定论文城市 × 五种策略。
- `cohort_strategies.csv`：按年度订单量汇总的论文样本整体结果，不外推全国。
- `report.json`：参数、来源、样本标识、分摊规则及与主实验的比较。

五种策略是全电动车、全无人机、排放最优、成本最优、时间最优。输出订单替代率、年排放、年运输运营成本、单均成本和平均运输时间。变化百分比相对全电动车，负值代表降低。

所有目标共享同一有效 OD 集合；异常行和已知排除订单量写入报告。无排除时要求排放基准与主实验一致，有排除时保留差异并标识不可直接等同比较。

验证只在 `hku-sto` 上运行本模块针对性测试及一次锁定样本分析，不重跑城市流水线或绘图。本地仅静态检查、commit、push。
