import argparse
import geopandas as gpd
from tqdm import tqdm


def remove_present_cultivation(all_cultivation_shapefile, present_cultivation_shapefile, 
                                    output_shapefile):
    """
    Directly remove present cultivation polygons from comprehensive shapefile.
    Assumes present cultivation polygons are exact subsets of the comprehensive dataset.
    
    Parameters:
    -----------
    all_cultivation_shapefile : str
        Path to shapefile containing all cultivation polygons (pre, post, present)
    present_cultivation_shapefile : str
        Path to shapefile containing only present cultivation polygons
    output_shapefile : str
        Path for output shapefile containing only pre and post cultivation polygons
    id_column : str, optional
        Column name to use for matching polygons (if None, uses geometry comparison)
    """
    
    print("Loading shapefiles...")
    
    # Load the comprehensive shapefile (all cultivation lands)
    all_cultivation_gdf = gpd.read_file(all_cultivation_shapefile)
    print(f"Loaded {len(all_cultivation_gdf)} polygons from comprehensive shapefile")
    
    # Load the present cultivation shapefile
    present_cultivation_gdf = gpd.read_file(present_cultivation_shapefile)
    print(f"Loaded {len(present_cultivation_gdf)} polygons from present cultivation shapefile")
    
    # Display column information
    print(f"\nAll cultivation columns: {list(all_cultivation_gdf.columns)}")
    print(f"Present cultivation columns: {list(present_cultivation_gdf.columns)}")
    
    
    # Ensure CRS compatibility
    if all_cultivation_gdf.crs != present_cultivation_gdf.crs:
        print("Reprojecting present cultivation shapefile to match comprehensive shapefile CRS...")
        present_cultivation_gdf = present_cultivation_gdf.to_crs(all_cultivation_gdf.crs)
    
    # Create set of present cultivation geometries (using WKT for comparison)
    print("Creating geometry hash for present cultivation polygons...")
    present_geoms = set()
    for geom in present_cultivation_gdf.geometry:
        try:
            # Use a simplified representation for comparison
            geom_wkt = geom.wkt
            present_geoms.add(geom_wkt)
        except Exception as e:
            print(f"Warning: Could not process geometry: {e}")
    
    print(f"Present cultivation unique geometries: {len(present_geoms)}")
    
    # Find polygons to keep (those not in present cultivation)
    polygons_to_keep = []
    removed_count = 0
    
    print("Comparing geometries...")
    for idx, row in all_cultivation_gdf.iterrows():
        try:
            geom_wkt = row.geometry.wkt
            
            if geom_wkt in present_geoms:
                removed_count += 1
                if removed_count <= 10:  # Show first 10 removals
                    print(f"Removing polygon {idx} (exact geometry match)")
            else:
                polygons_to_keep.append(idx)
                
        except Exception as e:
            print(f"Error processing polygon {idx}: {e}")
            # In case of error, keep the polygon
            polygons_to_keep.append(idx)


    for idx, row in tqdm(all_cultivation_gdf.iterrows(), total=len(all_cultivation_gdf), desc="Processing polygons"):
        try:
            geom_wkt = row.geometry.wkt

            if geom_wkt in present_geoms:
                removed_count += 1
            else:
                polygons_to_keep.append(idx)

        except Exception as e:
            print(f"Error processing polygon {idx}: {e}")
            polygons_to_keep.append(idx)
    
    # Create output GeoDataFrame
    output_gdf = all_cultivation_gdf.iloc[polygons_to_keep].copy()
    
    # Reset index and save
    output_gdf = output_gdf.reset_index(drop=True)
    output_gdf.to_file(output_shapefile)
    
    # Print results
    print(f"\n=== RESULTS ===")
    print(f"Original polygons (all cultivation): {len(all_cultivation_gdf)}")
    print(f"Present cultivation polygons: {len(present_cultivation_gdf)}")
    print(f"Removed polygons: {removed_count}")
    print(f"Remaining polygons (pre + post): {len(output_gdf)}")
    print(f"Output saved to: {output_shapefile}")
    
    # Verification
    expected_remaining = len(all_cultivation_gdf) - len(present_cultivation_gdf)

    if len(output_gdf) == expected_remaining:
        print(f"✓ Perfect match! Expected {expected_remaining}, got {len(output_gdf)}")
    else:
        print(f"⚠ Difference detected: Expected {expected_remaining}, got {len(output_gdf)}")
        print("This might indicate duplicate geometries or matching issues.")
    
    return len(output_gdf)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove present cultivation polygons from comprehensive cultivation dataset")
    
    parser.add_argument('--all_cultivation', required=True,
                        help='Path to shapefile containing all cultivation polygons (pre, post, present)')
    parser.add_argument('--present_cultivation', required=True,
                        help='Path to shapefile containing only present cultivation polygons')
    parser.add_argument('--output', required=True,
                        help='Path for output shapefile (pre + post cultivation only)')
    

    args = parser.parse_args()

    print(f"=== CULTIVATION POLYGON REMOVAL ===")


    result_count = remove_present_cultivation(
        args.all_cultivation,
        args.present_cultivation,
        args.output
    )


    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Final result: {result_count} polygons in output shapefile (pre + post cultivation)")
    
    if result_count == 0:
        print("WARNING: No polygons in output! Check your input data.")

    