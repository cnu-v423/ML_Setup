import geopandas as gpd
import pandas as pd
import argparse

def summarize_shapefile(shapefile_path):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Clean column names
    gdf.columns = [c.strip() for c in gdf.columns]

    # Column names
    village_col = "source_fol"    # Village name
    area_col = "polygon_ar"       # Area column (square meters)

    # Validations
    if village_col not in gdf.columns:
        raise ValueError(f"Column '{village_col}' not found in shapefile")

    if area_col not in gdf.columns:
        raise ValueError(f"Area column '{area_col}' not found in shapefile")

    # -----------------------------
    # Convert area to acres
    # -----------------------------
    SQM_TO_ACRES = 1 / 4046.8564224
    gdf["area_acres"] = gdf[area_col] * SQM_TO_ACRES

    # -----------------------------
    # Village-wise Summary
    # -----------------------------
    village_summary = (
        gdf.groupby(village_col)
        .agg(
            total_acres=("area_acres", "sum"),
            polygon_count=("area_acres", "count"),
            min_acres=("area_acres", "min"),
            max_acres=("area_acres", "max"),
            avg_acres=("area_acres", "mean")
        )
        .reset_index()
        .sort_values("total_acres", ascending=False)
    )

    # -----------------------------
    # Overall Summary
    # -----------------------------
    overall_summary = {
        "Total Villages": gdf[village_col].nunique(),
        "Total Polygons": len(gdf),
        "Total Acres": gdf["area_acres"].sum(),
        "Average Acres": gdf["area_acres"].mean(),
        "Minimum Acres": gdf["area_acres"].min(),
        "Maximum Acres": gdf["area_acres"].max(),
    }

    # -----------------------------
    # Print Nicely
    # -----------------------------
    print("\n=======================")
    print(" VILLAGE-WISE SUMMARY (ACRES)")
    print("=======================\n")
    print(village_summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=======================")
    print(" OVERALL SUMMARY (ACRES)")
    print("=======================\n")
    for k, v in overall_summary.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:,.2f}")
        else:
            print(f"{k:20}: {v}")

    return village_summary, overall_summary


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove present cultivation polygons from comprehensive cultivation dataset")
    
    parser.add_argument('--shapefile_path', required=True,
                        help='Path to shapefile containing all cultivation polygons (pre, post, present)')
    


    args = parser.parse_args()

    summarize_shapefile(args.shapefile_path)