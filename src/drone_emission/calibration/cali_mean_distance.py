import pandas as pd

MEAN_DISTANCE = 2.23  # in km

distance_nanshan_path = "/Users/linjingxian/Code/drone-emission/data/district/nanshan/backup_nanshan_od_pairs.csv"
nanshan_path = (
    "/Users/linjingxian/Code/drone-emission/data/district/nanshan/nanshan_od_pairs.csv"
)


def cali_mean_distance():
    distance_data = pd.read_csv(distance_nanshan_path)
    # print(distance_data.columns)

    raw_data = pd.read_csv(nanshan_path)
    data = raw_data[
        ["from_lat", "from_lng", "to_lat", "to_lng", "distance_km", "annual_sales"]
    ].copy()
    del raw_data
    # print(data.columns)

    # Merge road_distance_m from distance_data to data using (from_lat, from_lng, to_lat, to_lng) as key
    # Round coordinates to 6 decimal places (~0.1m precision) to handle floating point precision issues
    key_columns = ["from_lat", "from_lng", "to_lat", "to_lng"]
    round_decimals = 6

    # Create rounded key columns for matching
    for col in key_columns:
        data[f"{col}_round"] = data[col].round(round_decimals)
        distance_data[f"{col}_round"] = distance_data[col].round(round_decimals)

    round_key_columns = [f"{col}_round" for col in key_columns]

    data = data.merge(
        distance_data[round_key_columns + ["road_distance_m"]],
        on=round_key_columns,
        how="left",
    )

    # Drop the temporary rounded columns before saving
    data = data.drop(columns=round_key_columns)

    # save the data to csv
    data.to_csv(nanshan_path, index=False)
    print(f"Saved data to {nanshan_path}")

    print(data.head())

    # count for nan in road_distance_m
    nan_count = data["road_distance_m"].isna().sum()
    print(f"Number of NaN in road_distance_m: {nan_count}")

    # calculate the weighted mean of road_distance_m (weighted by annual_sales)
    weighted_mean = (data["road_distance_m"] * data["annual_sales"]).sum() / data[
        "annual_sales"
    ].sum()
    print(f"Weighted mean of road_distance_m: {weighted_mean}")


if __name__ == "__main__":
    cali_mean_distance()
