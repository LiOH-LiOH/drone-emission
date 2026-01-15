import os

import numpy as np
import pandas as pd

data_dir = "/Users/linjingxian/Code/drone-emission/data/raw/即时配送数据/疫情期间北京饿了么/eleme_round1_train_20200313/order/"


def analyze_coordinates():
    """分析订单数据的坐标范围，推测偏移量"""
    file_names = os.listdir(data_dir)
    file_paths = [
        os.path.join(data_dir, fn) for fn in file_names if fn.endswith(".txt")
    ]

    # 读取所有文件
    all_dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"总订单数: {len(combined)}")
    print(f"列名: {combined.columns.tolist()}")
    print()

    # 分析取货点（商家位置）坐标范围
    print("=== 取货点（商家）坐标范围 ===")
    print(
        f"pick_lng: {combined['pick_lng'].min():.6f} 到 {combined['pick_lng'].max():.6f}"
    )
    print(
        f"pick_lat: {combined['pick_lat'].min():.6f} 到 {combined['pick_lat'].max():.6f}"
    )
    print()

    # 分析配送点坐标范围
    print("=== 配送点坐标范围 ===")
    print(
        f"deliver_lng: {combined['deliver_lng'].min():.6f} 到 {combined['deliver_lng'].max():.6f}"
    )
    print(
        f"deliver_lat: {combined['deliver_lat'].min():.6f} 到 {combined['deliver_lat'].max():.6f}"
    )
    print()

    # 计算所有坐标的整体范围
    all_lng = pd.concat([combined["pick_lng"], combined["deliver_lng"]])
    all_lat = pd.concat([combined["pick_lat"], combined["deliver_lat"]])

    print("=== 所有坐标整体范围 ===")
    print(f"经度: {all_lng.min():.6f} 到 {all_lng.max():.6f}")
    print(f"纬度: {all_lat.min():.6f} 到 {all_lat.max():.6f}")
    print()

    # 计算数据中心点
    center_lng = all_lng.mean()
    center_lat = all_lat.mean()
    print(f"数据中心点: ({center_lng:.6f}, {center_lat:.6f})")
    print()

    # 北京市各区域参考坐标
    print("=== 北京市参考坐标 ===")
    beijing_refs = {
        "天安门": (116.397128, 39.916527),
        "朝阳区中心": (116.486409, 39.921489),
        "海淀区中心": (116.298056, 39.959912),
        "丰台区中心": (116.286968, 39.858247),
        "西城区中心": (116.365868, 39.912289),
        "东城区中心": (116.416357, 39.928353),
    }

    for name, (ref_lng, ref_lat) in beijing_refs.items():
        print(f"{name}: ({ref_lng:.6f}, {ref_lat:.6f})")
    print()

    # 推测偏移量（使用天安门作为参考）
    ref_lng, ref_lat = beijing_refs["天安门"]
    offset_lng = center_lng - ref_lng
    offset_lat = center_lat - ref_lat

    print("=== 推测偏移量（相对于天安门）===")
    print(f"经度偏移: {offset_lng:.6f}°")
    print(f"纬度偏移: {offset_lat:.6f}°")
    print()

    # 尝试反向校正并验证
    print("=== 反向校正后的坐标范围 ===")
    corrected_lng_min = all_lng.min() - offset_lng
    corrected_lng_max = all_lng.max() - offset_lng
    corrected_lat_min = all_lat.min() - offset_lat
    corrected_lat_max = all_lat.max() - offset_lat
    print(f"校正后经度: {corrected_lng_min:.6f} 到 {corrected_lng_max:.6f}")
    print(f"校正后纬度: {corrected_lat_min:.6f} 到 {corrected_lat_max:.6f}")
    print()

    # 分析可能的简单偏移模式
    print("=== 可能的简单偏移模式分析 ===")

    # 检查是否是简单的整数偏移
    simple_offsets = [5.0, 4.0, 3.0, 2.0, 1.0]
    for off in simple_offsets:
        test_lng = center_lng - off
        if 115.5 < test_lng < 117.5:  # 北京经度范围
            print(f"若经度偏移 {off}°, 则中心经度为 {test_lng:.6f}")

    print()

    # 检查纬度是否需要偏移
    if 39.4 < center_lat < 41.1:  # 北京纬度范围
        print(f"纬度 {center_lat:.6f} 在北京范围内，可能不需要纬度偏移")
    else:
        for off in [-1.0, -0.5, 0.5, 1.0]:
            test_lat = center_lat - off
            if 39.4 < test_lat < 41.1:
                print(f"若纬度偏移 {off}°, 则中心纬度为 {test_lat:.6f}")

    return combined, offset_lng, offset_lat


if __name__ == "__main__":
    analyze_coordinates()
    analyze_coordinates()
