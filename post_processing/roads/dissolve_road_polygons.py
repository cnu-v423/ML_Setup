import os
import os.path as osp
import glob
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Polygon, MultiPolygon


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
# VALUE_PATTERN = "*_dissolved.shp"
VALUE_PATTERN = '*_building.shp'
# "Anantavaram_10_classified_trees"


# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def load_shapefile(path, target_crs=None):
    print(f"📂 Loading: {path}")
    gdf = gpd.read_file(path)

    # Drop empty / null geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

    # Track source file (CRITICAL)
    gdf["src_file"] = osp.basename(path)

    # Reproject if needed
    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf

def find_all_value_2_shapefiles(base_folder):
    shp_files = []
    for root, _, _ in os.walk(base_folder):
        matches = glob.glob(os.path.join(root, VALUE_PATTERN))
        shp_files.extend(matches)

    if not shp_files:
        raise RuntimeError("❌ No *_value_2.shp files found")

    print(f"✅ Found {len(shp_files)} shapefiles")
    return shp_files
# def find_all_value_2_shapefiles(base_folder):
#     if not osp.isdir(base_folder):
#         raise ValueError(f"Invalid folder path: {base_folder}")

#     shp_files = [
#         osp.join(base_folder, f)
#         for f in os.listdir(base_folder)
#         if f.lower().endswith(".shp")
#     ]

#     if not shp_files:
#         raise RuntimeError("❌ No shapefiles found in folder")

#     print(f"✅ Found {len(shp_files)} shapefiles")

#     return shp_files


# ---------------------------------------------------------
# CORE LOGIC - COMPLETELY REWRITTEN
# ---------------------------------------------------------
def dissolve_only_cross_file_overlaps(gdf):
    """
    Dissolve polygons ONLY when they intersect with polygons
    from a DIFFERENT shapefile. Uses graph-based clustering.
    """

    print("🧠 Building spatial index")
    tree = STRtree(gdf.geometry.values)

    print("🔗 Building cross-file intersection graph")
    
    # Build adjacency graph: only connect polygons from different files
    adjacency = {i: set() for i in range(len(gdf))}
    
    for i in range(len(gdf)):
        geom_i = gdf.geometry.iloc[i]
        src_file_i = gdf.iloc[i]["src_file"]
        
        # Find candidates using spatial index
        candidate_indices = tree.query(geom_i, predicate='intersects')
        
        for j in candidate_indices:
            if i >= j:  # Avoid duplicate checks
                continue
            
            src_file_j = gdf.iloc[j]["src_file"]
            
            # ❗ CRITICAL: Only connect if from DIFFERENT files
            if src_file_i == src_file_j:
                continue
            
            geom_j = gdf.geometry.iloc[j]
            
            # Check if they truly intersect (not just touch at a point)
            if geom_i.intersects(geom_j):
                try:
                    intersection = geom_i.intersection(geom_j)
                    
                    # Must have area overlap (not just touching edges/points)
                    if intersection.area > 1e-10:  # Small threshold for floating point
                        adjacency[i].add(j)
                        adjacency[j].add(i)
                        print(f"  🔗 Cross-file link: {i} ({src_file_i}) ↔ {j} ({src_file_j})")
                except Exception as e:
                    print(f"  ⚠️ Geometry error between {i} and {j}: {e}")
                    continue

    print("🧩 Finding connected components (clusters to dissolve)")
    
    # Find connected components using DFS
    visited = set()
    clusters = []
    
    def dfs(node, cluster):
        """Depth-first search to find all connected polygons"""
        visited.add(node)
        cluster.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, cluster)
    
    for i in range(len(gdf)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)
    
    print(f"📊 Found {len(clusters)} clusters")
    
    # Build output geometries
    output_geoms = []
    
    for cluster in clusters:
        if len(cluster) > 1:
            # Check if cluster has cross-file polygons
            files_in_cluster = set(gdf.iloc[idx]["src_file"] for idx in cluster)
            if len(files_in_cluster) > 1:
                # Dissolve this cluster
                geoms_to_merge = [gdf.geometry.iloc[idx] for idx in cluster]
                merged_geom = unary_union(geoms_to_merge)
                output_geoms.append(merged_geom)
                print(f"  ✅ Dissolved cluster of {len(cluster)} polygons from {len(files_in_cluster)} files")
            else:
                # All from same file - keep separate
                for idx in cluster:
                    output_geoms.append(gdf.geometry.iloc[idx])
        else:
            # Single polygon with no cross-file connections
            output_geoms.append(gdf.geometry.iloc[cluster[0]])
    
    print("✅ Cross-file dissolve complete")
    return gpd.GeoDataFrame(geometry=output_geoms, crs=gdf.crs)


# ---------------------------------------------------------
# DIAGNOSTIC FUNCTION
# ---------------------------------------------------------
def diagnose_overlaps(gdf):
    """
    Diagnostic function to understand why polygons aren't merging.
    Call this to see what's happening with your data.
    """
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC MODE")
    print("="*60)
    
    tree = STRtree(gdf.geometry.values)
    
    for i in range(len(gdf)):
        geom_i = gdf.geometry.iloc[i]
        src_i = gdf.iloc[i]["src_file"]
        
        candidates = tree.query(geom_i, predicate='intersects')
        
        cross_file_intersections = []
        for j in candidates:
            if i == j:
                continue
            
            src_j = gdf.iloc[j]["src_file"]
            if src_i == src_j:
                continue
            
            geom_j = gdf.geometry.iloc[j]
            if geom_i.intersects(geom_j):
                intersection = geom_i.intersection(geom_j)
                cross_file_intersections.append({
                    'index': j,
                    'file': src_j,
                    'area': intersection.area,
                    'type': intersection.geom_type
                })
        
        if cross_file_intersections:
            print(f"\nPolygon {i} (from {src_i}):")
            for x in cross_file_intersections:
                print(f"  → Intersects polygon {x['index']} (from {x['file']})")
                print(f"    Intersection area: {x['area']:.6f}, Type: {x['type']}")


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def merge_road_polygons(input_folder, output_path, run_diagnostics=False):
    """
    Main function to merge road polygons.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing shapefiles
    output_path : str
        Path for output shapefile
    run_diagnostics : bool
        If True, run diagnostic analysis before processing
    """
    shp_files = find_all_value_2_shapefiles(input_folder)

    all_gdfs = []
    base_crs = None

    for shp in shp_files:
        gdf = load_shapefile(shp, base_crs)

        if gdf.empty:
            continue

        if base_crs is None:
            base_crs = gdf.crs

        all_gdfs.append(gdf)

    if not all_gdfs:
        raise RuntimeError("❌ No valid geometries found")

    print("🧩 Merging all shapefiles")
    merged = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True),
        crs=base_crs
    )

    print(f"📐 Total polygons before dissolve: {len(merged)}")
    print(f"📁 Files represented: {merged['src_file'].unique().tolist()}")
    
    if run_diagnostics:
        diagnose_overlaps(merged)
        print("\n" + "="*60)
        print("Diagnostics complete. Proceeding with merge...")
        print("="*60 + "\n")

    result_gdf = dissolve_only_cross_file_overlaps(merged)

    print(f"📐 Total polygons after dissolve: {len(result_gdf)}")

    print(f"💾 Writing output → {output_path}")
    result_gdf.to_file(output_path)

    print("\n🎉 DONE")
    print("✔ Polygons from same file: kept separate")
    print("✔ Polygons from different files with intersection: dissolved")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":

    # input_folder = (
    #     "/workspace/input/Predictions/Vegitation_predictions/"
    #     "post_process_shapefiles_after_resolved_tile_issue"
    # )

    # output_shapefile = (
    #     "/workspace/input/Predictions/Vegitation_predictions/"
    #     "post_process_shapefiles_after_resolved_tile_issue/"
    #     "merged_vegitation_cross_tile_only.shp"
    # )
    input_folder = '/workspace/input/Predictions/Roads_predictions/new_road_predictions/shape_files/new_shapefiles_with_smooth/Cleaned Finetuned Predictions'
    output_shapefile = '/workspace/input/Predictions/Roads_predictions/new_road_predictions/shape_files/new_shapefiles_with_smooth/Cleaned Finetuned Predictions/merged_road_polygons.shp'
    # Run with diagnostics to see what's happening
    merge_road_polygons(input_folder, output_shapefile, run_diagnostics=True)
    
    # Or run without diagnostics for faster processing:
    # merge_road_polygons(input_folder, output_shapefile, run_diagnostics=False)