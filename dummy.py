import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')




def remove_present_cultivation_direct(all_cultivation_shapefile, present_cultivation_shapefile, 
                                    output_shapefile, id_column=None):
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
    
    # Method 1: Remove by ID column if specified and available
    if id_column and id_column in all_cultivation_gdf.columns and id_column in present_cultivation_gdf.columns:
        print(f"\nUsing ID column '{id_column}' for matching...")
        
        present_ids = set(present_cultivation_gdf[id_column].values)
        print(f"Present cultivation IDs count: {len(present_ids)}")
        
        # Filter out present cultivation polygons
        pre_post_mask = ~all_cultivation_gdf[id_column].isin(present_ids)
        output_gdf = all_cultivation_gdf[pre_post_mask].copy()
        
        removed_count = len(all_cultivation_gdf) - len(output_gdf)
        print(f"Removed {removed_count} polygons using ID matching")
        
    else:
        # Method 2: Remove by geometry comparison
        print("\nUsing geometry comparison for matching...")
        
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


def remove_present_cultivation_by_index(all_cultivation_shapefile, present_cultivation_shapefile, 
                                      output_shapefile):
    """
    Alternative method: Remove by DataFrame index if the present cultivation 
    maintains the same index as the comprehensive dataset.
    """
    print("=== USING INDEX-BASED REMOVAL ===")
    
    all_cultivation_gdf = gpd.read_file(all_cultivation_shapefile)
    present_cultivation_gdf = gpd.read_file(present_cultivation_shapefile)
    
    print(f"All cultivation: {len(all_cultivation_gdf)} polygons")
    print(f"Present cultivation: {len(present_cultivation_gdf)} polygons")
    
    # Check if there's an index or FID column
    potential_id_cols = ['FID', 'OBJECTID', 'ID', 'index', 'ORIG_FID']
    
    for col in potential_id_cols:
        if col in all_cultivation_gdf.columns and col in present_cultivation_gdf.columns:
            print(f"Found potential ID column: {col}")
            return remove_present_cultivation_direct(
                all_cultivation_shapefile, 
                present_cultivation_shapefile, 
                output_shapefile, 
                id_column=col
            )
    
    # If no ID column found, use geometry comparison
    print("No suitable ID column found, using geometry comparison...")
    return remove_present_cultivation_direct(
        all_cultivation_shapefile, 
        present_cultivation_shapefile, 
        output_shapefile
    )


def verify_removal_results(all_cultivation_shapefile, present_cultivation_shapefile, 
                         output_shapefile):
    """
    Verify that the removal was successful by checking the results.
    """
    print("\n=== VERIFICATION ===")
    
    all_gdf = gpd.read_file(all_cultivation_shapefile)
    present_gdf = gpd.read_file(present_cultivation_shapefile)
    output_gdf = gpd.read_file(output_shapefile)
    
    print(f"Original total: {len(all_gdf)}")
    print(f"Present cultivation: {len(present_gdf)}")
    print(f"Remaining (pre + post): {len(output_gdf)}")
    print(f"Removed: {len(all_gdf) - len(output_gdf)}")
    
    # Check if the math adds up
    if len(output_gdf) + len(present_gdf) == len(all_gdf):
        print("✓ Perfect removal: All polygons accounted for")
    else:
        difference = len(all_gdf) - (len(output_gdf) + len(present_gdf))
        if difference > 0:
            print(f"⚠ {difference} polygons unaccounted for (possible duplicates in present cultivation)")
        else:
            print(f"⚠ {abs(difference)} extra polygons removed (possible overlap in datasets)")
    
    # Calculate areas if possible
    try:
        if all_gdf.crs and all_gdf.crs.to_string() != 'EPSG:4326':  # Not in degrees
            total_area = all_gdf.geometry.area.sum()
            present_area = present_gdf.geometry.area.sum() if len(present_gdf) > 0 else 0
            remaining_area = output_gdf.geometry.area.sum()
            
            print(f"\nArea verification:")
            print(f"Total area: {total_area:.2f}")
            print(f"Present cultivation area: {present_area:.2f}")
            print(f"Remaining area (pre + post): {remaining_area:.2f}")
            print(f"Area coverage: {(remaining_area/total_area*100):.1f}% remaining")
    except:
        print("Could not calculate area verification (CRS might be in degrees)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove present cultivation polygons from comprehensive cultivation dataset")
    
    parser.add_argument('--all_cultivation', required=True,
                        help='Path to shapefile containing all cultivation polygons (pre, post, present)')
    parser.add_argument('--present_cultivation', required=True,
                        help='Path to shapefile containing only present cultivation polygons')
    parser.add_argument('--output', required=True,
                        help='Path for output shapefile (pre + post cultivation only)')
    parser.add_argument('--id_column', type=str, default=None,
                        help='Column name to use for polygon matching (optional)')
    parser.add_argument('--method', choices=['direct', 'index'], default='direct',
                        help='Method: direct (geometry/ID comparison) or index (try ID columns first)')
    parser.add_argument('--verify', action='store_true',
                        help='Run verification after processing')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.all_cultivation):
        print(f"Error: All cultivation shapefile not found: {args.all_cultivation}")
        exit(1)
        
    if not os.path.exists(args.present_cultivation):
        print(f"Error: Present cultivation shapefile not found: {args.present_cultivation}")
        exit(1)
    
    print(f"=== DIRECT CULTIVATION POLYGON REMOVAL ===")
    print(f"Method: {args.method}")
    print(f"All cultivation shapefile: {args.all_cultivation}")
    print(f"Present cultivation shapefile: {args.present_cultivation}")
    print(f"Output shapefile: {args.output}")
    
    if args.method == 'direct':
        result_count = remove_present_cultivation_direct(
            args.all_cultivation,
            args.present_cultivation,
            args.output,
            id_column=args.id_column
        )
    elif args.method == 'index':
        result_count = remove_present_cultivation_by_index(
            args.all_cultivation,
            args.present_cultivation,
            args.output
        )
    
    # Run verification if requested
    if args.verify:
        verify_removal_results(args.all_cultivation, args.present_cultivation, args.output)
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Final result: {result_count} polygons in output shapefile (pre + post cultivation)")
    
    if result_count == 0:
        print("WARNING: No polygons in output! Check your input data.")