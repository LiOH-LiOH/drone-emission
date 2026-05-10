"""Entry point for rendering all paper figures.

Run:
    cd src/drone_emission
    ../../.venv/bin/python -m visualization.paper --png   # PNG (fast iteration)
    ../../.venv/bin/python -m visualization.paper          # PDF (for LaTeX)
"""

import argparse
import os
import sys

import pandas as pd


def _render_distance_comparison(data_dir, out, for_latex, plot_title, plot_fn):
    """Load Shanghai roadnet data and render the distance comparison figure."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")
    graphml_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_roadnet.graphml")

    if not os.path.exists(od_path) or not os.path.exists(graphml_path):
        print(f"Distance comparison skipped — missing data for {region_key}")
        return

    from roadnet.sparse_graph import SparseGraph
    from roadnet.tile_router import TileRouter

    od_df = pd.read_csv(od_path)
    graph = SparseGraph.from_graphml(graphml_path)
    router = TileRouter(graph)

    plot_fn(
        od_df=od_df,
        graph=graph,
        router=router,
        save_path=os.path.join(out, "distance_comparison.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def render_all(data_dir: str, figures_dir: str, for_latex: bool = False, plot_title: bool = False):
    from visualization.analysis import plot_province_orders_emission, plot_province_orders_both_emissions
    from visualization.concepts import plot_detour_ratio_concept, plot_emission_model_flowchart
    from visualization.roadnet import plot_distance_comparison

    out = os.path.join(figures_dir, "paper")
    os.makedirs(out, exist_ok=True)

    # --- Concept figures (no data dependency) ---
    plot_detour_ratio_concept(
        save_path=os.path.join(out, "detour_ratio_concept.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )

    plot_emission_model_flowchart(
        save_path=os.path.join(out, "emission_model_flowchart.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )

    # --- Data-driven figures ---
    stats_path = os.path.join(data_dir, "all_region_stats.csv")
    if not os.path.exists(stats_path):
        print(f"Missing {stats_path}; run consolidate_stats first — skipping data figures")
        return

    df = pd.read_csv(stats_path)

    plot_province_orders_emission(
        df,
        save_path=os.path.join(out, "province_orders_emission"),
        for_latex=for_latex,
        plot_title=plot_title,
    )

    plot_province_orders_both_emissions(
        df,
        save_path=os.path.join(out, "province_orders_both_emissions"),
        for_latex=for_latex,
        plot_title=plot_title,
    )

    _render_distance_comparison(
        data_dir=data_dir,
        out=out,
        for_latex=for_latex,
        plot_title=plot_title,
        plot_fn=plot_distance_comparison,
    )


def _bootstrap_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, ".."))
    if src not in sys.path:
        sys.path.insert(0, src)


def main():
    _bootstrap_paths()
    from utils import load_config

    parser = argparse.ArgumentParser(description="Render paper figures")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--png", action="store_true", help="Force PNG output")
    parser.add_argument("--titled", action="store_true", help="Add titles to figures")
    args = parser.parse_args()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    config = load_config(os.path.join(project_root, "experiment.toml"), env=args.env)
    for_latex = (not args.png) and config.get("for_latex", False)
    plot_title = args.titled or config.get("plot_title", False)

    print(f"for_latex={for_latex}, plot_title={plot_title}")
    render_all(config["data_dir"], config["figures_dir"], for_latex=for_latex, plot_title=plot_title)



if __name__ == "__main__":
    main()
