import geopandas as gpd
import pandas as pd
from pathlib import Path
import os
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
import numpy as np
from rtree import index
import warnings
import argparse
warnings.filterwarnings('ignore')

def merge_shapefiles(input_folder_path, output_path, overlap_threshold=0.1, method='spatial_index'):
    """
    Optimized merge of multiple shapefiles using spatial indexing for fast overlap detection.
    
    Parameters:
    - input_folder_path: Main directory containing subfolders with shapefiles
    - output_path: Path for the merged output shapefile
    - overlap_threshold: Minimum overlap ratio to consider polygons as overlapping
    - method: 'spatial_index' (fastest) or 'grid_based' (memory efficient)
    """
    
    # Find all shapefiles in subdirectories
    input_path = Path(input_folder_path)
    shapefile_paths = []
    
    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            shp_files = list(subfolder.glob("*.shp"))
            if shp_files:
                shapefile_paths.extend(shp_files)
                print(f"Found {len(shp_files)} shapefile(s) in folder: {subfolder.name}")
    
    if len(shapefile_paths) == 0:
        print("No shapefiles found in any subdirectories.")
        return
    
    print(f"\nTotal shapefiles found: {len(shapefile_paths)}")
    
    print("\nLoading shapefiles...")
    gdfs = []
    total_polygons = 0
    
    for i, shapefile_path in enumerate(shapefile_paths):
        try:
            gdf = gpd.read_file(shapefile_path)
            # Validate and fix geometries upfront
            gdf['geometry'] = gdf['geometry'].buffer(0)
            gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
            
            # Add metadata
            gdf['source_folder'] = shapefile_path.parent.name
            gdf['source_file'] = shapefile_path.name
            gdf['source_path'] = f"{shapefile_path.parent.name}/{shapefile_path.name}"
            gdf['file_priority'] = i
            gdf['global_id'] = range(total_polygons, total_polygons + len(gdf))
            
            gdfs.append(gdf)
            total_polygons += len(gdf)
            print(f"  Loaded {len(gdf)} polygons from {shapefile_path.parent.name}/{shapefile_path.name}")
            
        except Exception as e:
            print(f"  Error loading {shapefile_path.parent.name}/{shapefile_path.name}: {e}")
    
    if not gdfs:
        print("No valid shapefiles could be loaded.")
        return
    
    # Combine all GeoDataFrames
    print(f"\nCombining {total_polygons} polygons...")
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    combined_gdf = combined_gdf.sort_values('file_priority').reset_index(drop=True)
    
    print(f"Total polygons loaded: {len(combined_gdf)}")
    
    # Choose optimization method
    if method == 'spatial_index':
        final_gdf = _resolve_overlaps_spatial_index(combined_gdf, overlap_threshold)
    
    # Save result
    try:
        final_gdf.to_file(output_path)
        print(f"\nMerged shapefile saved to: {output_path}")
        
        # Print summary
        print(f"Final polygon count: {len(final_gdf)}")
        print(f"Polygons removed: {len(combined_gdf) - len(final_gdf)}")
        
        print("\nSummary by source:")
        summary = final_gdf.groupby('source_path').size().sort_values(ascending=False)
        for source, count in summary.items():
            print(f"  {source}: {count} polygons")
            
    except Exception as e:
        print(f"Error saving merged shapefile: {e}")

def _resolve_overlaps_spatial_index(gdf, overlap_threshold):
    """
    Fast overlap resolution using STRtree spatial index (O(n log n) complexity).
    """
    print("\nResolving overlaps using spatial index...")
    
    # Create spatial index
    geometries = gdf.geometry.tolist()
    spatial_index = STRtree(geometries)
    
    polygons_to_remove = set()
    processed_pairs = set()
    
    print("Building spatial index and detecting overlaps...")
    
    for i, geom in enumerate(geometries):
        if i in polygons_to_remove:
            continue
            
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(geometries)} polygons...")
        
        # Get potential overlaps using spatial index
        candidates = spatial_index.query(geom)
        
        for j in candidates:
            if j <= i or j in polygons_to_remove:
                continue
                
            pair_key = (min(i, j), max(i, j))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            try:
                other_geom = geometries[j]
                intersection = geom.intersection(other_geom)
                
                if not intersection.is_empty:
                    # Calculate overlap ratio
                    intersection_area = intersection.area
                    smaller_area = min(geom.area, other_geom.area)
                    
                    if smaller_area > 0:
                        overlap_ratio = intersection_area / smaller_area
                        
                        if overlap_ratio > overlap_threshold:
                            # Keep polygon from file with higher priority (lower file_priority number)
                            poly_i_priority = gdf.iloc[i]['file_priority']
                            poly_j_priority = gdf.iloc[j]['file_priority']
                            
                            if poly_i_priority <= poly_j_priority:
                                polygons_to_remove.add(j)
                            else:
                                polygons_to_remove.add(i)
                                break  # Current polygon is removed, no need to check more
                                
            except Exception as e:
                continue
    
    # Create final GeoDataFrame
    keep_indices = [i for i in range(len(gdf)) if i not in polygons_to_remove]
    final_gdf = gdf.iloc[keep_indices].copy().reset_index(drop=True)
    
    print(f"Removed {len(polygons_to_remove)} overlapping polygons")
    return final_gdf


def merge_shapefiles_by_area(input_folder_path, output_path, overlap_threshold=0.1, 
                           method='spatial_index', keep_larger_by_area=True, 
                           unique_largest_only=True):
    """
    Merge multiple shapefiles with area-based overlap resolution.
    
    Parameters:
    - input_folder_path: Main directory containing subfolders with shapefiles
    - output_path: Path for the merged output shapefile
    - overlap_threshold: Minimum overlap ratio to consider polygons as overlapping
    - method: 'spatial_index' (fastest) or 'grid_based' (memory efficient)
                          keep only the largest polygon once
    """

    input_path = Path(input_folder_path)
    shapefile_paths = []
    
    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            shp_files = list(subfolder.glob("*.shp"))
            if shp_files:
                shapefile_paths.extend(shp_files)
                print(f"Found {len(shp_files)} shapefile(s) in folder: {subfolder.name}")
    

    if len(shapefile_paths) == 0:
        print("No shapefiles found in any subdirectories.")
        return
    
    print(f"\nTotal shapefiles found: {len(shapefile_paths)}")
    
    print("\nLoading shapefiles...")
    gdfs = []
    total_polygons = 0

    for i, shapefile_path in enumerate(shapefile_paths):
        try:
            gdf = gpd.read_file(shapefile_path)
            # Validate and fix geometries upfront
            gdf['geometry'] = gdf['geometry'].buffer(0)
            gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
            
            # Add metadata
            gdf['source_folder'] = shapefile_path.parent.name
            gdf['source_file'] = shapefile_path.name
            gdf['source_path'] = f"{shapefile_path.parent.name}/{shapefile_path.name}"
            gdf['file_priority'] = i
            gdf['global_id'] = range(total_polygons, total_polygons + len(gdf))
            # Calculate and store area for each polygon
            gdf['polygon_area'] = gdf['geometry'].area
            
            gdfs.append(gdf)
            total_polygons += len(gdf)
            print(f"  Loaded {len(gdf)} polygons from {shapefile_path.parent.name}/{shapefile_path.name}")
            
        except Exception as e:
            print(f"  Error loading {shapefile_path.parent.name}/{shapefile_path.name}: {e}")
    
    if not gdfs:
        print("No valid shapefiles could be loaded.")
        return
    
    # Combine all GeoDataFrames
    print(f"\nCombining {total_polygons} polygons...")
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    combined_gdf = combined_gdf.sort_values('polygon_area', ascending=False).reset_index(drop=True)
    
    print(f"Total polygons loaded: {len(combined_gdf)}")
    print(f"Area-based processing: {'Enabled' if keep_larger_by_area else 'Disabled'}")
    print(f"Unique largest only: {'Enabled' if unique_largest_only else 'Disabled'}")
    
    # Choose optimization method
    if method == 'spatial_index':
        final_gdf = _resolve_overlaps_by_area_spatial_index(
            combined_gdf, overlap_threshold, keep_larger_by_area, unique_largest_only
        )
    
    # Save result
    try:
        final_gdf.to_file(output_path)
        print(f"\nMerged shapefile saved to: {output_path}")
        
        # Print summary
        print(f"Final polygon count: {len(final_gdf)}")
        print(f"Polygons removed: {len(combined_gdf) - len(final_gdf)}")
        
        print("\nSummary by source:")
        summary = final_gdf.groupby('source_path').size().sort_values(ascending=False)
        for source, count in summary.items():
            print(f"  {source}: {count} polygons")
            
        print(f"\nArea statistics:")
        print(f"  Total area retained: {final_gdf['polygon_area'].sum():.2f}")
        print(f"  Average polygon area: {final_gdf['polygon_area'].mean():.2f}")
        print(f"  Largest polygon area: {final_gdf['polygon_area'].max():.2f}")
        print(f"  Smallest polygon area: {final_gdf['polygon_area'].min():.2f}")
            
    except Exception as e:
        print(f"Error saving merged shapefile: {e}")


def _resolve_overlaps_by_area_spatial_index(gdf, overlap_threshold, keep_larger_by_area, unique_largest_only):
    """
    Area-based overlap resolution using STRtree spatial index.
    """
    print("\nResolving overlaps using spatial index (area-based)...")
    
    # Create spatial index
    geometries = gdf.geometry.tolist()
    spatial_index = STRtree(geometries)
    
    polygons_to_remove = set()
    processed_pairs = set()
    larger_polygon_groups = {}  # Track which smaller polygons belong to larger ones
    
    print("Building spatial index and detecting overlaps...")
    
    for i, geom in enumerate(geometries):
        if i in polygons_to_remove:
            continue
            
        if i % 2000 == 0:
            print(f"  Processed {i}/{len(geometries)} polygons...")
        
        # Get potential overlaps using spatial index
        candidates = spatial_index.query(geom)
        
        for j in candidates:
            if j <= i or j in polygons_to_remove:
                continue
                
            pair_key = (min(i, j), max(i, j))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            try:
                other_geom = geometries[j]
                intersection = geom.intersection(other_geom)
                
                if not intersection.is_empty:
                    # Calculate overlap ratio
                    intersection_area = intersection.area
                    smaller_area = min(geom.area, other_geom.area)
                    
                    if smaller_area > 0:
                        overlap_ratio = intersection_area / smaller_area
                        
                        if overlap_ratio > overlap_threshold:
                            # Get areas and folder information
                            area_i = gdf.iloc[i]['polygon_area']
                            area_j = gdf.iloc[j]['polygon_area']
                            folder_i = gdf.iloc[i]['source_folder']
                            folder_j = gdf.iloc[j]['source_folder']
                            
                            # Determine which polygon to keep
                            if keep_larger_by_area:
                                # Keep the larger polygon by area
                                if area_i > area_j:
                                    remove_idx = j
                                    keep_idx = i
                                elif area_j > area_i:
                                    remove_idx = i
                                    keep_idx = j
                                else:
                                    # If areas are equal, use original priority logic
                                    poly_i_priority = gdf.iloc[i]['file_priority']
                                    poly_j_priority = gdf.iloc[j]['file_priority']
                                    
                                    if poly_i_priority <= poly_j_priority:
                                        remove_idx = j
                                        keep_idx = i
                                    else:
                                        remove_idx = i
                                        keep_idx = j
                            else:
                                # Use original priority-based logic
                                poly_i_priority = gdf.iloc[i]['file_priority']
                                poly_j_priority = gdf.iloc[j]['file_priority']
                                
                                if poly_i_priority <= poly_j_priority:
                                    remove_idx = j
                                    keep_idx = i
                                else:
                                    remove_idx = i
                                    keep_idx = j
                            
                            # Handle unique_largest_only logic
                            if unique_largest_only and folder_i != folder_j:
                                # Track relationships for later processing
                                larger_idx = keep_idx if gdf.iloc[keep_idx]['polygon_area'] > gdf.iloc[remove_idx]['polygon_area'] else remove_idx
                                smaller_idx = remove_idx if larger_idx == keep_idx else keep_idx
                                
                                if larger_idx not in larger_polygon_groups:
                                    larger_polygon_groups[larger_idx] = []
                                larger_polygon_groups[larger_idx].append(smaller_idx)
                            
                            polygons_to_remove.add(remove_idx)
                            
                            if remove_idx == i:
                                break  # Current polygon is removed, no need to check more
                                
            except Exception as e:
                continue
    
    # Handle unique_largest_only processing
    if unique_largest_only:
        print("Processing unique largest only logic...")
        for larger_idx, smaller_indices in larger_polygon_groups.items():
            if larger_idx in polygons_to_remove:
                continue
                
            # Among all smaller polygons that overlap with this larger one,
            # keep only the largest one
            valid_smaller = [idx for idx in smaller_indices if idx not in polygons_to_remove]
            
            if len(valid_smaller) > 1:
                # Find the largest among the smaller polygons
                largest_smaller_idx = max(valid_smaller, key=lambda x: gdf.iloc[x]['polygon_area'])
                
                # Remove all others except the largest
                for idx in valid_smaller:
                    if idx != largest_smaller_idx:
                        polygons_to_remove.add(idx)
    
    # Create final GeoDataFrame
    keep_indices = [i for i in range(len(gdf)) if i not in polygons_to_remove]
    final_gdf = gdf.iloc[keep_indices].copy().reset_index(drop=True)
    
    print(f"Removed {len(polygons_to_remove)} overlapping polygons")
    return final_gdf



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract building polygons with straight lines')
    parser.add_argument('--input_folder_path', required=True, help='Input shapefile folder')
    parser.add_argument('--filter_by_area',action='store_true', default=False, help='Input shapefile folder')

    
    args = parser.parse_args()

    # input_folder_path = "/workspace/output/building_detections/predictions/shapefiles"
    
    if(args.filter_by_area):

        merge_shapefiles_by_area(
            input_folder_path=args.input_folder_path,
            output_path=args.input_folder_path + "/cultivation_predictions_merged_shape_file.shp",
            overlap_threshold=0.1,
            method='spatial_index',
        )

    else:
        merge_shapefiles(
            input_folder_path=args.input_folder_path,
            output_path= args.input_folder_path + "/building_predictions_merged_shape_file.shp",
            overlap_threshold=0.1,
            method='spatial_index'
        )