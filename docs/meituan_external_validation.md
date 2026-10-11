# 美团保定：北京参数的外部验证

本流程只验证已经接受的北京饿了么参数，不重新拟合，不更新
`experiment.toml` 或 `gravity_calibration_summary.json`。
默认参数来源是已接受的校准摘要，**不是**数值模块中的历史
`CURRENT_PARAMETERS` 常量。报告记录参数摘要、坐标恢复、订单和建筑输入的哈希。

## 输入与规则

- 使用 `data/meituan_restored_20261011` 中已经恢复坐标的订单包；读取明确的
  WGS84 坐标列，核验恢复记录中的订单文件 SHA256。
- 仅保留 `is_courier_grabbed == 1`，共 568,546 个唯一订单；排除 85,797 条拒单。
- 时间范围为 2022-10-17 至 2022-10-24。全部数据均为外部验证集，没有训练集；
  默认报告全部有效订单的总体结果，`--by-date` 可另加每日结果。
  每日子集不是训练/测试划分。
- 4,962 个商家对应 5,065 个实际取餐位置。多个取餐位置分别建立候选集，
  不用商家的首个或众数位置替换订单坐标。空间诊断以取餐位置为单位。
- 复用北京流程：目的地首先匹配覆盖建筑，否则匹配 50 m 内最近建筑边界；
  只保留体积有效为正、实际目的地属于取餐点 5 km 候选集的订单。
- 对完整候选集比较固定北京参数与均匀建筑选择。报告条件似然、排名、
  精确距离分布误差、取餐位置空间拟合，以及每一步排除数量。

## 建筑输入

需要完整 GBA 瓦片，而非仅保定行政区建筑文件。订单涉及城区、清苑和容城；
行政边界裁剪会截断 5 km 候选集。准备命令从全部取餐点外扩 5.1 km，并覆盖
全部收货点的匹配缓冲范围。当前需 `e115_n40_e120_n35` 瓦片。

```bash
PYTHONPATH=src .venv/bin/python -m drone_emission --env local calibrate-external-buildings \
  --bundle data/meituan_restored_20261011 \
  --gba-tiles gba_data/LoD1_GeoJSON/asiaeast/e115_n40_e120_n35.gpkg \
  --output data/calibration/meituan_external/inputs/baoding_buildings.gpkg
```

建筑体积沿用现有 `demand.cal_area.preprocess_buildings`：EPSG:3857 面积乘高度。
这保留了与北京拟合的输入口径，但 Web Mercator 面积具有纬度相关畸变，
不应解释为独立测量的真实体积。建筑日期与 2022 年订单日期也可能不同。
准备过程生成建筑文件和 `.coverage.json`；外部验证强制核对覆盖记录与输入哈希。

## 固定参数验证

```bash
PYTHONPATH=src .venv/bin/python -m drone_emission --env local calibrate-external \
  --bundle data/meituan_restored_20261011 \
  --buildings data/calibration/meituan_external/inputs/baoding_buildings.gpkg \
  --output data/calibration/meituan_external/runs \
  --memory-limit-gb 12
```

结果写入以输入哈希命名的子目录：`external_validation.json`、
`model_metrics.csv`、`distance_distribution_source.csv.gz`、
`pickup_location_spatial_fit.csv.gz` 和准备缓存。没有参数接受或配置覆盖选项。
每完成一个模型/日期范围就保存带输入哈希和分布文件哈希的检查点，
并更新 `model_metrics.csv`。相同输入重跑可恢复已经完成的评分。
只有生成 `external_validation.json` 才表示全部评估完成；中途 CSV 是部分结果。
远端长任务应通过持久 SSH 会话以 `nohup` 后台运行，并记录 PID、日志和退出状态，
使连接丢失不会终止计算。

本机物理内存 24 GiB。准备阶段根据输入文件大小估计内存，候选缓存使用 mmap；
准备完成后额外按每候选对 256 字节估计精确全局距离分布评估的峰值。
超过给定限制则停止并保留缓存，应改到高内存远端运行；这只是保守估算，不是峰值保证。

判断能否迁移参数时，应先看匹配保留比例，再看相对于均匀选择的改进及距离误差。
北京 2020 年与保定 2022 年存在城市、平台、时期差异；坐标恢复也仍有原始坐标系
独立认证不足的限制。外部验证失败不等于坐标恢复失败，也不能仅凭 POI 对齐率
认定重力模型有效。
