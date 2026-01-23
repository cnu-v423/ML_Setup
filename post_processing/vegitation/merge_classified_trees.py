import geopandas as gpd
from shapely.ops import unary_union
from shapely.strtree import STRtree

## ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
INPUT_SHP = "/workspace/input/Predictions/Vegitation_predictions/sam3_predictions/shapefiles/Mangalagiri_1024_overlap_512/Mangalagiri_classified_trees.shp"
OUTPUT_SHP = "/workspace/input/Predictions/Vegitation_predictions/sam3_predictions/shapefiles/Mangalagiri_1024_overlap_512/Mangalagiri_merged_trees.shp"

VALID_CLASSES = [2, 3, 4]   # shrubs + trees only
MIN_AREA = 1.0             # m²

# ------------------------------------------------------------
# LOAD & CLEAN
# ------------------------------------------------------------
gdf = gpd.read_file(INPUT_SHP)
    
if gdf.crs.is_geographic:
    raise ValueError("CRS must be projected (meters)")

gdf = gdf[gdf["class_id"].isin(VALID_CLASSES)].copy()
gdf["geometry"] = gdf.geometry.buffer(0)
gdf = gdf[gdf.geometry.area >= MIN_AREA].reset_index(drop=True)

# ------------------------------------------------------------
# BUILD STRTREE WITH INDEX MAP
# ------------------------------------------------------------
geoms = gdf.geometry.tolist()
geom_id_to_idx = {id(g): i for i, g in enumerate(geoms)}
tree = STRtree(geoms)

visited = set()
components = []

# ------------------------------------------------------------
# FIND TOUCHING COMPONENTS (NO INTERSECTION REQUIRED)
# ------------------------------------------------------------
for i, geom in enumerate(geoms):
    if i in visited:
        continue

    stack = [i]
    component = []

    while stack:
        idx = stack.pop()
        if idx in visited:
            continue

        visited.add(idx)
        component.append(idx)

        for cand in tree.query(geoms[idx]):
            j = geom_id_to_idx.get(id(cand))
            if j is None or j in visited:
                continue

            # ✅ TOUCHING ONLY (NOT OVERLAP)
            if geoms[idx].touches(cand):
                stack.append(j)

    components.append(component)

print(f"🌳 Detected {len(components)} trees")

# ------------------------------------------------------------
# DOMINANT CLASS COLLAPSE (KEY FIX)
# ------------------------------------------------------------
rows = []

for comp in components:
    subset = gdf.iloc[comp]

    # dominant height class
    dominant_class = int(subset["class_id"].max())
    dominant_name = subset.loc[
        subset["class_id"] == dominant_class, "class_name"
    ].iloc[0]

    # 🔴 KEEP ONLY DOMINANT CLASS GEOMETRY
    dominant_geoms = subset.loc[
        subset["class_id"] == dominant_class, "geometry"
    ]

    final_geom = unary_union(dominant_geoms)

    rows.append({
        "geometry": final_geom,
        "class_id": dominant_class,
        "class_name": dominant_name,
        "area_sqm": final_geom.area
    })

# ------------------------------------------------------------
# SAVE FINAL RESULT
# ------------------------------------------------------------
final_gdf = gpd.GeoDataFrame(rows, crs=gdf.crs)
final_gdf = final_gdf[final_gdf.geometry.area >= MIN_AREA]

final_gdf.to_file(OUTPUT_SHP)

print("✅ Dominant-class tree merging completed")
print(f"📁 Output: {OUTPUT_SHP}")
print(f"🌳 Final tree count: {len(final_gdf)}")
