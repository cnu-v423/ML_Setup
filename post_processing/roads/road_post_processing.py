from osgeo import gdal, ogr, osr
import os
from pathlib import Path

# ---------------- CONFIG ----------------
input_folder = "/media/vassardigitalgpu/One Touch/Predictions/Roads_predictions/new_road_predictions/without_prob"
output_folder = "/media/vassardigitalgpu/One Touch/Predictions/Roads_predictions/new_road_predictions/shape_files/new_shapefiles_with_smooth"

SIMPLIFY_TOLERANCE = 0.8     # smaller = safer
BUFFER_DISTANCE = 0.6       # must be > half road width
SNAP_TOLERANCE = 0.2

os.makedirs(output_folder, exist_ok=True)

tiff_files = list(Path(input_folder).glob("*.tif")) + list(Path(input_folder).glob("*.tiff"))
print(f"Found {len(tiff_files)} TIFFs")

# ---------------------------------------
def fix_geom(geom):
    """Robust geometry fixer"""
    if geom is None or geom.IsEmpty():
        return None

    geom = geom.MakeValid()
    geom = geom.Buffer(0)           # clean rings
    geom = geom.SimplifyPreserveTopology(SIMPLIFY_TOLERANCE)

    if geom.GetGeometryType() not in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
        return None

    if geom.GetGeometryType() == ogr.wkbPolygon:
        mp = ogr.Geometry(ogr.wkbMultiPolygon)
        mp.AddGeometry(geom)
        geom = mp

    return geom


# ---------------------------------------
for tiff_file in tiff_files:
    print(f"\nProcessing: {tiff_file.name}")

    src_ds = gdal.Open(str(tiff_file))
    band = src_ds.GetRasterBand(1)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())

    base = tiff_file.stem
    shp_dir = os.path.join(output_folder, base)
    os.makedirs(shp_dir, exist_ok=True)

    shp_path = os.path.join(shp_dir, f"{base}.shp")

    drv = ogr.GetDriverByName("ESRI Shapefile")
    drv.DeleteDataSource(shp_path)
    ds = drv.CreateDataSource(shp_path)
    layer = ds.CreateLayer("roads", srs, ogr.wkbMultiPolygon)

    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

    # ---- Polygonize into memory ----
    mem_drv = ogr.GetDriverByName("Memory")
    mem_ds = mem_drv.CreateDataSource("")
    tmp = mem_ds.CreateLayer("tmp", srs, ogr.wkbPolygon)
    tmp.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

    gdal.Polygonize(band, None, tmp, 0)

    # ---- Collect road geometries ----
    geoms = []
    for f in tmp:
        if f.GetField("id") == 1:
            g = f.GetGeometryRef()
            if g:
                geoms.append(g.Clone())

    if not geoms:
        print("⚠ No road pixels found")
        continue

    # ---- Dissolve ALL touching roads ----
    dissolved = ogr.Geometry(ogr.wkbMultiPolygon)
    for g in geoms:
        dissolved = dissolved.Union(g)

    # ---- Fix + smooth ----
    dissolved = fix_geom(dissolved)

    # ---- Smooth safely ----
    dissolved = dissolved.Buffer(BUFFER_DISTANCE)
    dissolved = dissolved.Buffer(-BUFFER_DISTANCE)

    # ---- Final cleanup ----
    dissolved = fix_geom(dissolved)

    # ---- Save features ----
    for i in range(dissolved.GetGeometryCount()):
        poly = dissolved.GetGeometryRef(i)
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("id", 1)
        feat.SetGeometry(poly)
        layer.CreateFeature(feat)
        feat = None

    ds = None
    src_ds = None

    print("✓ Saved valid geometry")

print("\n✅ ALL FILES PROCESSED — QGIS SAFE")

# from osgeo import gdal, ogr, osr
# import os
# from pathlib import Path

# # Input folder containing GeoTIFF files
# input_folder = "/media/vassardigitalgpu/One Touch/Predictions/Roads_predictions/new_road_predictions/without_prob"
# # Output folder for shapefiles
# output_folder = "/media/vassardigitalgpu/One Touch/Predictions/Roads_predictions/new_road_predictions/shape_files/new_shapefiles_with_smooth"

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Get all TIFF files from input folder
# tiff_files = list(Path(input_folder).glob("*.tif")) + list(Path(input_folder).glob("*.tiff"))
# print(f"Found {len(tiff_files)} TIFF file(s) to process")

# # Smoothing parameters
# SIMPLIFY_TOLERANCE = 1.0  # Adjust this value (higher = more simplified)
# BUFFER_DISTANCE = 0.5     # Smooth by buffering (positive then negative)

# # Process each TIFF file
# for tiff_file in tiff_files:
#     input_raster = str(tiff_file)
    
#     # Create a separate folder for this shapefile
#     base_name = tiff_file.stem
#     shapefile_folder = os.path.join(output_folder, base_name)
#     os.makedirs(shapefile_folder, exist_ok=True)
    
#     # Create output shapefile path inside the dedicated folder
#     output_shapefile = os.path.join(shapefile_folder, f"{base_name}.shp")
    
#     print(f"\nProcessing: {tiff_file.name}")
#     print(f"  Output folder: {shapefile_folder}")
    
#     try:
#         # Open raster
#         src_ds = gdal.Open(input_raster)
#         if src_ds is None:
#             print(f"  ERROR: Could not open {input_raster}")
#             continue
            
#         srcband = src_ds.GetRasterBand(1)
        
#         # Get projection from raster
#         srs = osr.SpatialReference()
#         srs.ImportFromWkt(src_ds.GetProjection())
        
#         # Create shapefile
#         driver = ogr.GetDriverByName("ESRI Shapefile")
#         driver.DeleteDataSource(output_shapefile)  # remove if exists
#         out_ds = driver.CreateDataSource(output_shapefile)
#         out_layer = out_ds.CreateLayer("roads", srs=srs, geom_type=ogr.wkbPolygon)
        
#         # Add ID field
#         field_defn = ogr.FieldDefn("id", ogr.OFTInteger)
#         out_layer.CreateField(field_defn)
        
#         # # Temporary full polygonization
#         # tmp_layer = out_ds.CreateLayer("tmp", srs=srs, geom_type=ogr.wkbPolygon)
#         # tmp_layer.CreateField(field_defn)
#         # gdal.Polygonize(srcband, None, tmp_layer, 0, [], callback=None)

#         # Temporary in-memory layer (NO tmp.shp will be created)
#         mem_driver = ogr.GetDriverByName("Memory")
#         mem_ds = mem_driver.CreateDataSource("mem")
#         tmp_layer = mem_ds.CreateLayer("tmp", srs=srs, geom_type=ogr.wkbPolygon)
#         tmp_layer.CreateField(field_defn)

#         gdal.Polygonize(srcband, None, tmp_layer, 0, [], callback=None)
        
#         # Copy and smooth only road polygons (value == 1) into final layer
#         road_count = 0
#         for feature in tmp_layer:
#             if feature.GetField("id") == 1:   # Only road pixels
#                 geom = feature.GetGeometryRef()
                
#                 if geom is not None:
#                     # Apply smoothing operations
                    
#                     # 1. Simplify to reduce vertices (Douglas-Peucker algorithm)
#                     smoothed_geom = geom.Simplify(SIMPLIFY_TOLERANCE)
                    
#                     # 2. Buffer operation for additional smoothing
#                     # Positive buffer then negative buffer (Chaikin's algorithm equivalent)
#                     smoothed_geom = smoothed_geom.Buffer(BUFFER_DISTANCE)
#                     smoothed_geom = smoothed_geom.Buffer(-BUFFER_DISTANCE)
                    
#                     # Create new feature with smoothed geometry
#                     new_feature = ogr.Feature(out_layer.GetLayerDefn())
#                     new_feature.SetGeometry(smoothed_geom)
#                     new_feature.SetField("id", 1)
#                     out_layer.CreateFeature(new_feature)
#                     new_feature = None
#                     road_count += 1
        
#         # Cleanup
#         out_ds = None
#         src_ds = None
        
#         print(f"  ✓ Saved: {output_shapefile}")
#         print(f"  Road polygons extracted and smoothed: {road_count}")
        
#         # List all files created in the folder
#         created_files = list(Path(shapefile_folder).glob(f"{base_name}.*"))
#         print(f"  Files created: {', '.join([f.suffix for f in created_files])}")
        
#     except Exception as e:
#         print(f"  ERROR processing {tiff_file.name}: {str(e)}")
#         continue

# print(f"\n{'='*50}")
# print(f"Processing complete!")
# print(f"Shapefiles saved to: {output_folder}")
# print(f"\nSmoothing parameters used:")
# print(f"  - Simplify tolerance: {SIMPLIFY_TOLERANCE}")
# print(f"  - Buffer distance: {BUFFER_DISTANCE}")