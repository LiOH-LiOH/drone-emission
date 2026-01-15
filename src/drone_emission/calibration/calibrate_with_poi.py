"""使用POI地标物校准坐标偏移量"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

poi_dir = "/Users/linjingxian/Code/drone-emission/data/raw/即时配送数据/北京市poi202209"
elm_data_dir = "/Users/linjingxian/Code/drone-emission/data/raw/即时配送数据/疫情期间北京饿了么/eleme_round1_train_20200313/order/"

# 知名地标的真实WGS84坐标（来源：高德/百度地图）
LANDMARKS = {
    "天安门": (116.397390, 39.908860),
    "故宫博物院": (116.403414, 39.924091),
    "北京站": (116.427163, 39.902624),
    "北京西站": (116.322056, 39.894539),
    "北京南站": (116.378888, 39.865036),
    "首都机场": (116.603039, 40.079531),
    "北京大学": (116.310905, 39.992806),
    "清华大学": (116.326836, 40.003530),
    "中关村": (116.316833, 39.981949),
    "王府井": (116.410612, 39.914587),
    "西单": (116.374115, 39.912134),
    "国贸": (116.461040, 39.909340),
    "三里屯": (116.454303, 39.936478),
    "鸟巢": (116.395610, 39.992997),
    "颐和园": (116.275175, 39.999934),
}


def load_poi_data():
    """加载风景名胜POI数据"""
    shp_path = os.path.join(poi_dir, "北京市_风景名胜_202209.shp")
    gdf = gpd.read_file(shp_path)
    print(f"风景名胜POI数量: {len(gdf)}")
    print(f"列名: {gdf.columns.tolist()}")
    print(f"CRS: {gdf.crs}")
    return gdf


def load_transport_poi():
    """加载交通设施POI数据"""
    shp_path = os.path.join(poi_dir, "北京市_交通设施服务_202209.shp")
    gdf = gpd.read_file(shp_path)
    print(f"交通设施POI数量: {len(gdf)}")
    return gdf


def load_edu_poi():
    """加载科教文化POI数据"""
    shp_path = os.path.join(poi_dir, "北京市_科教文化服务_202209.shp")
    gdf = gpd.read_file(shp_path)
    print(f"科教文化POI数量: {len(gdf)}")
    return gdf


def search_landmark(gdf, keyword):
    """在POI数据中搜索地标"""
    # 尝试不同的列名
    name_cols = ["name", "NAME", "名称", "POI名称"]
    for col in name_cols:
        if col in gdf.columns:
            matches = gdf[gdf[col].str.contains(keyword, na=False)]
            if len(matches) > 0:
                return matches, col
    return None, None


def calculate_offset(poi_coord, real_coord):
    """计算偏移量"""
    offset_lng = poi_coord[0] - real_coord[0]
    offset_lat = poi_coord[1] - real_coord[1]
    return offset_lng, offset_lat


def main():
    # 加载多个POI数据集
    print("=== 加载POI数据 ===")
    scenic_gdf = load_poi_data()
    transport_gdf = load_transport_poi()
    edu_gdf = load_edu_poi()

    print()
    print("=== 查看数据示例 ===")
    print(scenic_gdf.head())
    print()

    # 合并所有POI
    all_gdf = pd.concat([scenic_gdf, transport_gdf, edu_gdf], ignore_index=True)
    print(f"总POI数量: {len(all_gdf)}")

    # 搜索地标并计算偏移
    print()
    print("=== 地标匹配与偏移计算 ===")
    offsets = []

    for landmark_name, real_coord in LANDMARKS.items():
        matches, name_col = search_landmark(all_gdf, landmark_name)
        if matches is not None and len(matches) > 0:
            # 取第一个匹配结果
            match = matches.iloc[0]
            poi_lng = match.geometry.x
            poi_lat = match.geometry.y

            offset_lng, offset_lat = calculate_offset((poi_lng, poi_lat), real_coord)
            offsets.append(
                {
                    "landmark": landmark_name,
                    "poi_lng": poi_lng,
                    "poi_lat": poi_lat,
                    "real_lng": real_coord[0],
                    "real_lat": real_coord[1],
                    "offset_lng": offset_lng,
                    "offset_lat": offset_lat,
                }
            )
            print(f"{landmark_name}:")
            print(f"  POI坐标: ({poi_lng:.6f}, {poi_lat:.6f})")
            print(f"  真实坐标: ({real_coord[0]:.6f}, {real_coord[1]:.6f})")
            print(f"  偏移量: (lng: {offset_lng:.6f}, lat: {offset_lat:.6f})")
            print()

    if offsets:
        # 计算平均偏移量
        df = pd.DataFrame(offsets)

        # 过滤异常值（偏移量超过0.02度的视为异常）
        df_filtered = df[
            (df["offset_lng"].abs() < 0.02) & (df["offset_lat"].abs() < 0.02)
        ]

        print("=== 过滤后的地标偏移量 ===")
        for _, row in df_filtered.iterrows():
            print(
                f"{row['landmark']}: lng偏移={row['offset_lng']:.6f}, lat偏移={row['offset_lat']:.6f}"
            )
        print()

        mean_offset_lng = df_filtered["offset_lng"].mean()
        mean_offset_lat = df_filtered["offset_lat"].mean()
        std_offset_lng = df_filtered["offset_lng"].std()
        std_offset_lat = df_filtered["offset_lat"].std()

        print("=== 偏移量统计（过滤异常后）===")
        print(f"有效地标数量: {len(df_filtered)}")
        print(f"平均经度偏移: {mean_offset_lng:.6f}° (标准差: {std_offset_lng:.6f})")
        print(f"平均纬度偏移: {mean_offset_lat:.6f}° (标准差: {std_offset_lat:.6f})")
        print()
        print("=== POI数据校正值（POI -> 真实坐标）===")
        print(f"POI_OFFSET_LNG = {-mean_offset_lng:.6f}")
        print(f"POI_OFFSET_LAT = {-mean_offset_lat:.6f}")
        print()

        # POI坐标本身是GCJ02坐标系，偏移很小
        # 饿了么数据的偏移需要单独分析
        # 这里POI偏移主要是GCJ02和WGS84之间的差异（约0.006度）

        return -mean_offset_lng, -mean_offset_lat
    else:
        print("未找到匹配的地标")
        return None, None


if __name__ == "__main__":
    main()
    main()
