"""通过匹配饿了么数据与POI数据来确定精确偏移量"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

poi_dir = "/Users/linjingxian/Code/drone-emission/data/raw/即时配送数据/北京市poi202209"
elm_data_dir = "/Users/linjingxian/Code/drone-emission/data/raw/即时配送数据/疫情期间北京饿了么/eleme_round1_train_20200313/order/"


def load_elm_shops():
    """加载饿了么商家位置"""
    file_names = os.listdir(elm_data_dir)
    file_paths = [
        os.path.join(elm_data_dir, fn) for fn in file_names if fn.endswith(".txt")
    ]

    all_dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # 获取唯一商家位置
    shops = (
        combined.groupby("shop_id")
        .agg({"pick_lng": "first", "pick_lat": "first"})
        .reset_index()
    )

    print(f"饿了么商家数量: {len(shops)}")
    print(
        f"商家经度范围: {shops['pick_lng'].min():.6f} 到 {shops['pick_lng'].max():.6f}"
    )
    print(
        f"商家纬度范围: {shops['pick_lat'].min():.6f} 到 {shops['pick_lat'].max():.6f}"
    )

    return shops


def load_poi_restaurants():
    """加载餐饮服务POI"""
    shp_path = os.path.join(poi_dir, "北京市_餐饮服务_202209.shp")
    gdf = gpd.read_file(shp_path)
    print(f"餐饮服务POI数量: {len(gdf)}")
    print(f"POI经度范围: {gdf.geometry.x.min():.6f} 到 {gdf.geometry.x.max():.6f}")
    print(f"POI纬度范围: {gdf.geometry.y.min():.6f} 到 {gdf.geometry.y.max():.6f}")
    return gdf


def grid_search_offset(
    elm_shops, poi_coords, offset_lng_range, offset_lat_range, step=0.01
):
    """网格搜索最佳偏移量"""
    best_offset = None
    best_matches = 0
    best_avg_dist = float("inf")

    # 构建POI的KDTree
    poi_tree = cKDTree(poi_coords)

    results = []

    for off_lng in np.arange(offset_lng_range[0], offset_lng_range[1], step):
        for off_lat in np.arange(offset_lat_range[0], offset_lat_range[1], step):
            # 应用偏移
            adjusted_lng = elm_shops["pick_lng"].values + off_lng
            adjusted_lat = elm_shops["pick_lat"].values + off_lat
            adjusted_coords = np.column_stack([adjusted_lng, adjusted_lat])

            # 查找最近邻
            distances, indices = poi_tree.query(adjusted_coords, k=1)

            # 统计在阈值内的匹配数量（0.001度约等于100米）
            threshold = 0.001
            matches = np.sum(distances < threshold)
            avg_dist = np.mean(distances)

            results.append(
                {
                    "offset_lng": off_lng,
                    "offset_lat": off_lat,
                    "matches": matches,
                    "avg_dist": avg_dist,
                }
            )

            if matches > best_matches or (
                matches == best_matches and avg_dist < best_avg_dist
            ):
                best_matches = matches
                best_avg_dist = avg_dist
                best_offset = (off_lng, off_lat)

    return best_offset, best_matches, best_avg_dist, pd.DataFrame(results)


def main():
    print("=== 加载数据 ===")
    elm_shops = load_elm_shops()
    poi_gdf = load_poi_restaurants()

    # 提取POI坐标
    poi_coords = np.column_stack([poi_gdf.geometry.x, poi_gdf.geometry.y])

    print()
    print("=== 网格搜索偏移量 ===")
    print("搜索范围: 经度 [-5.5, -4.5], 纬度 [0.5, 1.5]")
    print("步长: 0.05度")

    # 粗搜索
    best_offset, matches, avg_dist, results = grid_search_offset(
        elm_shops,
        poi_coords,
        offset_lng_range=(-5.5, -4.5),
        offset_lat_range=(0.5, 1.5),
        step=0.05,
    )

    print(f"\n粗搜索最佳偏移: lng={best_offset[0]:.4f}, lat={best_offset[1]:.4f}")
    print(f"匹配商家数: {matches}, 平均距离: {avg_dist:.6f}度")

    # 细搜索
    print("\n精细搜索...")
    best_offset_fine, matches_fine, avg_dist_fine, results_fine = grid_search_offset(
        elm_shops,
        poi_coords,
        offset_lng_range=(best_offset[0] - 0.1, best_offset[0] + 0.1),
        offset_lat_range=(best_offset[1] - 0.1, best_offset[1] + 0.1),
        step=0.01,
    )

    print(
        f"精细搜索最佳偏移: lng={best_offset_fine[0]:.4f}, lat={best_offset_fine[1]:.4f}"
    )
    print(f"匹配商家数: {matches_fine}, 平均距离: {avg_dist_fine:.6f}度")

    # 更精细搜索
    print("\n超精细搜索...")
    best_offset_ultra, matches_ultra, avg_dist_ultra, _ = grid_search_offset(
        elm_shops,
        poi_coords,
        offset_lng_range=(best_offset_fine[0] - 0.02, best_offset_fine[0] + 0.02),
        offset_lat_range=(best_offset_fine[1] - 0.02, best_offset_fine[1] + 0.02),
        step=0.001,
    )

    print(
        f"超精细搜索最佳偏移: lng={best_offset_ultra[0]:.6f}, lat={best_offset_ultra[1]:.6f}"
    )
    print(f"匹配商家数: {matches_ultra}, 平均距离: {avg_dist_ultra:.6f}度")

    # 极精细搜索
    print("\n极精细搜索（步长0.0001度≈11米）...")
    best_offset_final, matches_final, avg_dist_final, _ = grid_search_offset(
        elm_shops,
        poi_coords,
        offset_lng_range=(best_offset_ultra[0] - 0.005, best_offset_ultra[0] + 0.005),
        offset_lat_range=(best_offset_ultra[1] - 0.005, best_offset_ultra[1] + 0.005),
        step=0.0001,
    )

    print(
        f"极精细搜索最佳偏移: lng={best_offset_final[0]:.6f}, lat={best_offset_final[1]:.6f}"
    )
    print(f"匹配商家数: {matches_final}, 平均距离: {avg_dist_final:.6f}度")
    print(f"平均距离约: {avg_dist_final * 111000:.1f}米")

    print()
    print("=== 推荐校正值 ===")
    print(f"OFFSET_LNG = {best_offset_final[0]:.6f}")
    print(f"OFFSET_LAT = {best_offset_final[1]:.6f}")

    return best_offset_final


if __name__ == "__main__":
    main()
    main()
