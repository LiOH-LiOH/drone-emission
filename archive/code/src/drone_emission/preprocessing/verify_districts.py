"""验证 regions.json 中的区县名称是否存在于 combined_data 中"""

import json

import pandas as pd

# 加载数据
regions_path = "/Users/linjingxian/Code/drone-emission/regions.json"
data_path = "/Users/linjingxian/Code/drone-emission/data/combined_data.csv"

with open(regions_path, "r", encoding="utf-8") as f:
    regions = json.load(f)

df = pd.read_csv(data_path)
available_districts = set(df["district"].dropna().unique())

print(f"combined_data 中共有 {len(available_districts)} 个区县")
print(f"regions.json 中共有 {len(regions)} 个区域配置\n")

print("=" * 60)
print(f"{'区域key':<12} {'中文名':<10} {'是否存在':<8} {'数据条数'}")
print("=" * 60)

missing = []
found = []

for key, config in regions.items():
    district_cn = config["region_name_cn"]
    exists = district_cn in available_districts

    if exists:
        count = len(df[df["district"] == district_cn])
        found.append((key, district_cn, count))
        print(f"{key:<12} {district_cn:<10} ✓        {count}")
    else:
        missing.append((key, district_cn))
        print(f"{key:<12} {district_cn:<10} ✗        -")

print("=" * 60)
print(f"\n找到: {len(found)}/{len(regions)}")

if missing:
    print(f"\n缺失的区县 ({len(missing)}):")
    for key, name in missing:
        print(f"  - {key}: {name}")

    # 尝试模糊匹配
    print("\n可能的匹配:")
    for key, name in missing:
        # 去掉"区"后缀匹配
        name_base = name.replace("区", "").replace("市", "")
        matches = [d for d in available_districts if name_base in d]
        if matches:
            print(f"  {name} -> {matches}")
