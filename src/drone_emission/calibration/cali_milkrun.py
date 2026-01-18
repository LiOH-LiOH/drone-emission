import pandas as pd

eg_path = "/Users/linjingxian/Code/drone-emission/data/raw/Milkrun_data/order level data until 20211021.csv"


def cali_milkrun_data(file_path):
    milkrun_df = pd.read_csv(file_path)
    print(f"Total orders in Milkrun data: {len(milkrun_df)}")
    # print(milkrun_df.head())

    # count for nan in "delivery_lat", "delivery_lng", and "delivery_distance"
    nan_lat_count = milkrun_df["delivery_lat"].isna().sum()
    nan_lng_count = milkrun_df["delivery_lng"].isna().sum()
    nan_distance_count = milkrun_df["delivery_distance"].isna().sum()
    print(f"Number of NaN in delivery_lat: {nan_lat_count}")
    print(f"Number of NaN in delivery_lng: {nan_lng_count}")
    print(f"Number of NaN in delivery_distance: {nan_distance_count}")


if __name__ == "__main__":
    cali_milkrun_data(eg_path)
