import os
import geopandas as gpd


def preprocess_buildings(building_path, save_path):
    """
    Preprocess building data: calculate area * height (volume) for each building.
    Save the preprocessed data to the data folder.
    """
    buildings_gdf = gpd.read_file(building_path)
    print(f"Loaded {len(buildings_gdf)} buildings from {building_path}")

    # === Calculate area (in EPSG:3857, unit is meters) ===
    # Ensure we're in a projected CRS for accurate area calculation
    if buildings_gdf.crs and buildings_gdf.crs.to_epsg() != 3857:
        buildings_gdf_proj = buildings_gdf.to_crs(epsg=3857)
    else:
        buildings_gdf_proj = buildings_gdf

    # Calculate area in square meters
    buildings_gdf["area_m2"] = buildings_gdf_proj.geometry.area

    # === Calculate volume (area * height) ===
    buildings_gdf["volume_m3"] = buildings_gdf["area_m2"] * buildings_gdf["height"]

    # === Save preprocessed data ===
    buildings_gdf.to_file(save_path, driver="GeoJSON")
    print(f"Saved preprocessed buildings to {save_path}")

    # Print summary statistics
    print(f"\n=== Building Preprocessing Summary ===")
    print(f"Total buildings: {len(buildings_gdf)}")
    print(
        f"Area (m²): min={buildings_gdf['area_m2'].min():.1f}, "
        f"median={buildings_gdf['area_m2'].median():.1f}, "
        f"max={buildings_gdf['area_m2'].max():.1f}"
    )
    print(
        f"Height (m): min={buildings_gdf['height'].min():.1f}, "
        f"median={buildings_gdf['height'].median():.1f}, "
        f"max={buildings_gdf['height'].max():.1f}"
    )
    print(
        f"Volume (m³): min={buildings_gdf['volume_m3'].min():.1f}, "
        f"median={buildings_gdf['volume_m3'].median():.1f}, "
        f"max={buildings_gdf['volume_m3'].max():.1f}"
    )

    return buildings_gdf
