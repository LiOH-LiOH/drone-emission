import pandas as pd


PADDING = 0.05


def calculate_bounds(city_df):
    print("-" * 20)
    print(
        f"  Lat: min={city_df['lat'].min():.4f}, 0.1%={city_df['lat'].quantile(0.001):.4f}, 50%={city_df['lat'].median():.4f}, 99.9%={city_df['lat'].quantile(0.999):.4f}, max={city_df['lat'].max():.4f}"
    )
    print(
        f"  Lng: min={city_df['lng'].min():.4f}, 0.1%={city_df['lng'].quantile(0.001):.4f}, 50%={city_df['lng'].median():.4f}, 99.9%={city_df['lng'].quantile(0.999):.4f}, max={city_df['lng'].max():.4f}"
    )
    print("-" * 20)

    return {
        "min_lat": city_df["lat"].quantile(0.001) - PADDING,
        "max_lat": city_df["lat"].quantile(0.999) + PADDING,
        "min_lng": city_df["lng"].quantile(0.001) - PADDING,
        "max_lng": city_df["lng"].quantile(0.999) + PADDING,
    }


def filter_city_data(df, city_name_cn, boundaries_dict=None):
    tmp_data = df[df["city"] == city_name_cn]
    print(f"{city_name_cn} has {len(tmp_data)} entries.")

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

    print(f"After filtering, {city_name_cn} has {len(valid_data)} valid entries.\n")
    return valid_data


def read_city_data(data_path, city_name_cn, boundaries_dict=None):
    df = pd.read_csv(data_path)
    city_data = filter_city_data(df, city_name_cn, boundaries_dict)
    return city_data
