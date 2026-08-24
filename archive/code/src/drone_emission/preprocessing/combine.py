import pandas as pd
import os


def main():
    all_data = []

    for id in range(1, 6):
        print(f"Processing file {id}...")
        data = pd.read_excel(
            f"/Users/linjingxian/Code/drone-emission/data/raw/美团网全国各省市区县外卖商家商户明细数据202108含配送时间区域电话等等/全国外卖数据(省市区县)output_{id}.xlsx"
        )

        use_cols = [
            "id",
            "province",
            "city",
            "district",
            "month_sales",
            "lat",
            "lng",
            "hours",
        ]
        data_use = data[use_cols].copy()
        data_use["month_sales"] = (
            data_use["month_sales"].astype(str).str.extract(r"(\d+)").astype(float)
        )

        data_use = data_use.dropna(subset=["month_sales"])
        data_use = data_use[data_use["month_sales"] != 0]

        all_data.append(data_use)

    final_df = pd.concat(all_data, ignore_index=True)
    output_path = "/Users/linjingxian/Code/drone-emission/data/combined_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved combined data to {output_path}")
    final_df.head()


if __name__ == "__main__":
    main()
