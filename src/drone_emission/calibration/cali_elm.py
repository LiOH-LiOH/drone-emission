import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import contextily as ctx
from utils.coordinates import gcj02_to_wgs84
from visualization._utils import configure_contextily_cache, finalize_plot, setup_figure

elm_data_dir = "/Users/linjingxian/Code/drone-emission/data/raw/即时配送数据/疫情期间北京饿了么/eleme_round1_train_20200313/order/"


def get_eg_data():
    # get file names in elm_data_dir
    file_names = os.listdir(elm_data_dir)
    file_paths = [
        os.path.join(elm_data_dir, fn) for fn in file_names if fn.endswith(".txt")
    ]

    eg_df = pd.read_csv(file_paths[0])
    # find the shod_id appearing most frequently and its location
    eg_id = eg_df["shop_id"].value_counts().idxmax()
    eg_shop_loc = (
        eg_df[eg_df["shop_id"] == eg_id][["pick_lng", "pick_lat"]].iloc[0].tolist()
    )

    # from all files, extract orders from this shop_id, only save deliver_lng, deliver_lat
    all_orders = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        shop_orders = df[df["shop_id"] == eg_id][["deliver_lng", "deliver_lat"]]
        all_orders.append(shop_orders)

    return eg_id, eg_shop_loc, pd.concat(all_orders, ignore_index=True)


def caliberate_elm_data():
    id, loc, orders = get_eg_data()
    print(f"Total orders from shop {id}: {len(orders)}")
    print(f"Shop location: {loc}")
    print(orders.head())

    min_lng, min_lat = orders[["deliver_lng", "deliver_lat"]].min().tolist()
    max_lng, max_lat = orders[["deliver_lng", "deliver_lat"]].max().tolist()
    print(f"Delivery location bounds:")
    print(f"Longitude: {min_lng} to {max_lng}")
    print(f"Latitude: {min_lat} to {max_lat}")


def plot_delivery_coverage(save_path=None):
    """绘制商家位置和配送范围散点图，带有地图底图"""
    shop_id, shop_loc, orders = get_eg_data()

    # 数据偏移修正（通过POI匹配确定的精确偏移量）
    # 原始数据被偏移：经度+5.1192°，纬度-0.781°
    # 匹配率97.0%（11267/11633商家），平均误差约28.8米
    OFFSET_LNG = -5.1192
    OFFSET_LAT = 0.781

    # 提取配送点坐标并修正偏移
    deliver_lng = orders["deliver_lng"].values + OFFSET_LNG
    deliver_lat = orders["deliver_lat"].values + OFFSET_LAT

    # 修正商家坐标偏移
    shop_lng_corrected = shop_loc[0] + OFFSET_LNG
    shop_lat_corrected = shop_loc[1] + OFFSET_LAT

    # 将 GCJ02 坐标转换为 WGS84
    shop_lng_wgs, shop_lat_wgs = gcj02_to_wgs84(shop_lng_corrected, shop_lat_corrected)

    # 转换配送点坐标
    deliver_lng_wgs = []
    deliver_lat_wgs = []
    for lng, lat in zip(deliver_lng, deliver_lat):
        w_lng, w_lat = gcj02_to_wgs84(lng, lat)
        deliver_lng_wgs.append(w_lng)
        deliver_lat_wgs.append(w_lat)
    deliver_lng_wgs = np.array(deliver_lng_wgs)
    deliver_lat_wgs = np.array(deliver_lat_wgs)

    # 计算边界（包含商家位置和所有配送点）
    all_lng = np.append(deliver_lng_wgs, shop_lng_wgs)
    all_lat = np.append(deliver_lat_wgs, shop_lat_wgs)

    # 添加padding
    padding = 0.005
    boundaries_dict = {
        "min_lng": all_lng.min() - padding,
        "max_lng": all_lng.max() + padding,
        "min_lat": all_lat.min() - padding,
        "max_lat": all_lat.max() + padding,
    }

    # 设置图形
    fig, ax, aspect_ratio, _, _ = setup_figure(all_lng, all_lat, boundaries_dict)

    # 绘制配送点散点图
    sc = ax.scatter(
        deliver_lng_wgs,
        deliver_lat_wgs,
        c="blue",
        s=10,
        alpha=0.5,
        zorder=10,
        edgecolors="none",
        label=f"Delivery Points ({len(orders)})",
    )

    # 绘制商家位置（用红色大点标记）
    ax.scatter(
        shop_lng_wgs,
        shop_lat_wgs,
        c="red",
        s=200,
        marker="*",
        zorder=20,
        edgecolors="black",
        linewidth=1,
        label=f"Shop (ID: {shop_id})",
    )

    # 添加地图底图（不绘制行政边界）
    configure_contextily_cache()
    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom=14,
        alpha=0.6,
        zorder=0,
    )

    # 添加图例
    ax.legend(loc="upper right", fontsize=10)

    # 统计信息
    stats_text = (
        f"Shop ID: {shop_id}\n"
        f"Total Orders: {len(orders)}\n"
        f"Shop Location:\n"
        f"  Lng: {shop_lng_wgs:.6f}\n"
        f"  Lat: {shop_lat_wgs:.6f}"
    )

    finalize_plot(
        fig,
        ax,
        save_path,
        f"Eleme Shop {shop_id} Delivery Coverage",
        stats_text,
        boundaries_dict,
    )


if __name__ == "__main__":
    caliberate_elm_data()
    # 保存图片到 figures/calibration 目录
    save_path = "figures/calibration/delivery_coverage.png"
    plot_delivery_coverage(save_path=save_path)
