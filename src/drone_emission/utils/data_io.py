import pandas as pd

PADDING = 0.05


def calculate_bounds(city_df):
    print("-" * 20)
    print(
        f"  Lat: min={city_df['lat'].min():.4f}, 1%={city_df['lat'].quantile(0.01):.4f}, 50%={city_df['lat'].median():.4f}, 99%={city_df['lat'].quantile(0.99):.4f}, max={city_df['lat'].max():.4f}"
    )
    print(
        f"  Lng: min={city_df['lng'].min():.4f}, 1%={city_df['lng'].quantile(0.01):.4f}, 50%={city_df['lng'].median():.4f}, 99%={city_df['lng'].quantile(0.99):.4f}, max={city_df['lng'].max():.4f}"
    )
    print("-" * 20)

    return {
        "min_lat": city_df["lat"].quantile(0.01) - PADDING,
        "max_lat": city_df["lat"].quantile(0.99) + PADDING,
        "min_lng": city_df["lng"].quantile(0.01) - PADDING,
        "max_lng": city_df["lng"].quantile(0.99) + PADDING,
    }


def filter_region_data(
    data_path, region_name, boundaries_dict=None, region_level="city", save_path=None
):
    df = pd.read_csv(data_path)
    print(f"Total entries in dataset: {len(df)}")

    tmp_data = df[df[region_level] == region_name]
    print(f"{region_name} has {len(tmp_data)} entries.")

    if boundaries_dict is None:
        boundaries_dict = calculate_bounds(tmp_data)

    min_lat = boundaries_dict["min_lat"]
    max_lat = boundaries_dict["max_lat"]
    min_lng = boundaries_dict["min_lng"]
    max_lng = boundaries_dict["max_lng"]

    print(f"Using hardcoded boundaries:")
    print(f"  Latitude: {min_lat} to {max_lat}")
    print(f"  Longitude: {min_lng} to {max_lng}")

    valid_data = tmp_data[
        tmp_data["lat"].notna()
        & tmp_data["lng"].notna()
        & tmp_data["month_sales"].notna()
        & (tmp_data["lat"] >= min_lat)
        & (tmp_data["lat"] <= max_lat)
        & (tmp_data["lng"] >= min_lng)
        & (tmp_data["lng"] <= max_lng)
    ].copy()

    print(f"After filtering, {region_name} has {len(valid_data)} valid entries.\n")

    if save_path:
        valid_data.to_csv(save_path, index=False)
        print(f"Saved filtered data to {save_path}\n")
