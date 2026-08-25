"""Formal observed distance renderers used only by the manifest registry."""

from contextlib import contextmanager
import json
import os
import pickle
import time

import pandas as pd


def _render_observed_delivery_distance_distribution(
    data_dir,
    out,
    for_latex,
    plot_title,
):
    """Render the full observed Eleme delivery-distance distribution."""
    order_dir = os.path.join(
        data_dir,
        "raw",
        "即时配送数据",
        "疫情期间北京饿了么",
        "eleme_round1_train_20200313",
        "order",
    )
    from drone_emission.calibration import cali_elm
    from drone_emission.visualization.observed_distance_plot import (
        plot_observed_delivery_distance_distribution,
    )

    distance_summary = None
    if os.path.isdir(order_dir):
        cali_elm.elm_data_dir = order_dir + os.sep
        distances, sample_metadata = cali_elm.load_all_eleme_distances(
            min_orders_per_shop=None,
            max_distance_km=None,
            return_metadata=True,
        )
    else:
        import numpy as np

        locked_path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "calibration",
                "observed_delivery_distance_histogram.csv",
            )
        )
        locked = pd.read_csv(locked_path)
        distances = np.repeat(
            locked["center_km"].to_numpy(dtype=float),
            locked["count"].to_numpy(dtype=int),
        )
        sample_metadata = {
            "source": "locked Beijing OD histogram",
            "source_file_count": 29,
            "raw_order_count": 278733,
            "raw_shop_count": 11633,
            "min_orders_per_shop": None,
            "orders_after_shop_filter": 278733,
            "shops_after_shop_filter": 11633,
            "finite_positive_distance_count": 278688,
            "excluded_invalid_or_zero_count": 45,
            "max_distance_km": None,
            "above_max_distance_count": 0,
            "retained_distance_count": 278688,
        }
        distance_summary = {
            "count": 278688,
            "min": 0.00011119062365690736,
            "q1": 0.7110052863443016,
            "median": 1.2393072588493563,
            "mean": 1.3523227505534225,
            "q3": 1.8804782161212,
            "max": 8.18385784959984,
        }
        print(
            "observed_delivery_distance_distribution: using locked "
            f"histogram {locked_path}"
        )
    plot_observed_delivery_distance_distribution(
        distances,
        save_path=os.path.join(
            out,
            "observed_delivery_distance_distribution.png",
        ),
        for_latex=for_latex,
        plot_title=plot_title,
        sample_metadata=sample_metadata,
        distance_summary=distance_summary,
    )
