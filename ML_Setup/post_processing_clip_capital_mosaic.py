#!/usr/bin/env python3
"""
Enhanced Geospatial Processing Script
This script processes predicted cultivation lands with capital city mosaic data
to identify unauthorized cultivation areas and RS.NO plots.
"""

import geopandas as gpd
import pandas as pd
import uuid
import os
import argparse
import numpy as np


def validate_inputs(inputs):
    """Validate that input files exist and output directory is accessible"""
    print("\n🔍 VALIDATING INPUTS...")
    
    # Check input files
    required_files = ['predicted_cultivation_shapefile', 'capital_city_mosaic_shapefile', 'returnable_plots_shapefile']
    for key in required_files:
        if not os.path.exists(inputs[key]):
            raise FileNotFoundError(f"Input file not found: {inputs[key]}")
    
    # Create output directory if it doesn't exist
    os.makedirs(inputs['output_dir'], exist_ok=True)
    
    print("✅ All inputs validated successfully!")


def result_2_clip_cultivation_with_mosaic(inputs):
    """
    Result 2: Clip capital city mosaic polygons with predicted cultivation lands shapefile.
    Get exact predicted cultivation lands with parent properties of RS.NO polygons.
    """
    print("\n" + "=" * 60)
    print("RESULT 2: CLIPPING CULTIVATION LANDS WITH CAPITAL CITY MOSAIC")
    print("=" * 60)
    
    # Load shapefiles
    cultivation_gdf = gpd.read_file(inputs['predicted_cultivation_shapefile'])
    mosaic_gdf = gpd.read_file(inputs['capital_city_mosaic_shapefile'])
    
    
    # Ensure same CRS
    if cultivation_gdf.crs != mosaic_gdf.crs:
        print("Reprojecting cultivation data to match mosaic CRS...")
        cultivation_gdf = cultivation_gdf.to_crs(mosaic_gdf.crs)
    
    # Spatial join: assign parent mosaic polygon attributes to each predicted polygon
    result_2 = gpd.sjoin(
        cultivation_gdf, 
        mosaic_gdf, 
        how="left",          # keep all predicted polygons
        predicate="intersects"  # or "within", depending on your data
    )

    # Drop unnecessary spatial join columns (like index_right)
    result_2 = result_2.drop(columns=["index_right"], errors="ignore")

    # Reproject to UTM for area calculation
    if result_2.crs.is_geographic:
        result_2 = result_2.to_crs(f"EPSG:{inputs['utm_epsg']}")
    
    # Calculate area in acres
    result_2["cultivation_area_ac"] = result_2.geometry.area / 4046.85642
    
    # Generate UUIDs
    result_2["cultivation_uuid"] = [str(uuid.uuid4()) for _ in range(len(result_2))]
    
    # Save Result 2
    result_2_path = os.path.join(inputs['output_dir'], "result_2_cultivation_with_mosaic_properties.shp")
    result_2.to_file(result_2_path)
    
    print(f"💾 Saved Result 2 to:")
    print(f"   - {result_2_path}")
    
    return result_2, result_2_path


def result_3_extract_rsno_polygons(inputs, result_2):
    """
    Result 3: Extract only the capital city polygons (mosaic) 
    where at least one predicted polygon exists.
    """
    print("\n" + "=" * 60)
    print("RESULT 3: EXTRACTING CAPITAL CITY POLYGONS WITH PREDICTIONS")
    print("=" * 60)

    # Load capital city mosaic
    mosaic_gdf = gpd.read_file(inputs['capital_city_mosaic_shapefile'])

    # Ensure CRS match
    if mosaic_gdf.crs != result_2.crs:
        print("Reprojecting mosaic data to match result_2 CRS...")
        mosaic_gdf = mosaic_gdf.to_crs(result_2.crs)

    # Spatial join: assign mosaic polygon attributes to predicted polygons
    joined = gpd.sjoin(result_2, mosaic_gdf, how="inner", predicate="intersects")

    if joined.empty:
        print("No overlaps found between predictions and mosaic polygons.")
        return None, None

    # Get unique capital mosaic polygons that intersect with predictions
    print("Extracting unique mosaic polygons with predictions...")
    unique_ids = joined.index_right.unique()   # index of mosaic polygons in original file
    capital_with_predictions = mosaic_gdf.loc[unique_ids].copy()

    print(f"Found {len(capital_with_predictions)} capital city polygons with predictions.")

    # Save result
    result_3_path = os.path.join(inputs['output_dir'], "result_3_capital_with_predictions.shp")
    capital_with_predictions.to_file(result_3_path)

    print(f"💾 Saved Result 3 to:\n   - {result_3_path}")

    return capital_with_predictions, result_3_path



def result_4_unauthorized_cultivation_lands(inputs, result_2):
    """
    Result 4: Filter Result 2 by removing returnable plots and category equal to land acquisition
    to get unauthorized cultivation land polygons.
    """
    print("\n" + "=" * 60)
    print("RESULT 4: IDENTIFYING UNAUTHORIZED CULTIVATION LANDS")
    print("=" * 60)
    
    # Load returnable plots
    print("📥 Loading returnable plots...")
    returnable_gdf = gpd.read_file(inputs['returnable_plots_shapefile'])
    
    # Ensure same CRS
    if returnable_gdf.crs != result_2.crs:
        print("🔄 Reprojecting returnable plots to match result_2 CRS...")
        returnable_gdf = returnable_gdf.to_crs(result_2.crs)
    
    # Create unified returnable geometry
    print("🔗 Creating unified returnable geometry...")
    returnable_union = returnable_gdf.union_all()
    
    # Remove cultivation areas that intersect with returnable plots
    print("✂️  Removing cultivation areas intersecting with returnable plots...")
    result_2_copy = result_2.copy()
    
    # Check for intersections with returnable plots
    intersects_returnable = result_2_copy.geometry.intersects(returnable_union)
    result_2_filtered = result_2_copy[~intersects_returnable].copy()
    
    print(f"🗑️  Removed {len(result_2) - len(result_2_filtered)} cultivation areas intersecting with returnable plots")
    
    # Filter out category equal to land acquisition
    exclude_category = inputs.get('exclude_category', 'Land Acquisition')
    
    # Find category column
    category_column = None
    for col in result_2_filtered.columns:
        if 'category' in col.lower():
            category_column = col
            break
    
    if category_column:
        print(f"🔍 Filtering out category: '{exclude_category}' from column '{category_column}'...")
        initial_count = len(result_2_filtered)
        result_4 = result_2_filtered[result_2_filtered[category_column] != exclude_category].copy()
        final_count = len(result_4)
        print(f"🗑️  Removed {initial_count - final_count} areas with category '{exclude_category}'")
    else:
        print("⚠️  Warning: Category column not found. Skipping category filtering.")
        result_4 = result_2_filtered.copy()
    
    # Generate unauthorized cultivation UUIDs
    result_4["unauthorized_cultivation_uuid"] = [str(uuid.uuid4()) for _ in range(len(result_4))]
    
    print(f"✅ Generated {len(result_4)} unauthorized cultivation polygons")
    
    # Save Result 4
    result_4_path = os.path.join(inputs['output_dir'], "result_4_unauthorized_cultivation_lands.shp")
    result_4_geojson = os.path.join(inputs['output_dir'], "result_4_unauthorized_cultivation_lands.geojson")
    result_4_csv = os.path.join(inputs['output_dir'], "result_4_unauthorized_cultivation_lands.csv")
    
    result_4.to_file(result_4_path)
    result_4.to_file(result_4_geojson, driver="GeoJSON")
    
    # Create CSV version
    result_4_csv_data = result_4.drop(columns=['geometry']).copy()
    result_4_csv_data.to_csv(result_4_csv, index=False)
    
    print(f"💾 Saved Result 4 to:")
    print(f"   - {result_4_path}")
    print(f"   - {result_4_geojson}")
    print(f"   - {result_4_csv}")
    
    return result_4, result_4_path


def result_5_unauthorized_rsno_plots(inputs, result_3, result_4):
    """
    Result 5: Get unauthorized RS.NO plots from capital city mosaic shapefile.
    If any RS.NO plot has at least one unauthorized cultivation, consider that RS.NO plot 
    as unauthorized cultivation RS.NO plot.
    """
    print("\n" + "=" * 60)
    print("RESULT 5: IDENTIFYING UNAUTHORIZED RS.NO PLOTS")
    print("=" * 60)
    
    # Get RS.NO column name
    rs_column = None
    for col in result_3.columns:
        if 'rs' in col.lower() and 'no' in col.lower():
            rs_column = col
            break
    
    if not rs_column:
        possible_rs_cols = ['rs_no', 'rsno', 'RS_NO', 'RSNO', 'rs.no', 'RS.NO']
        for col in possible_rs_cols:
            if col in result_3.columns:
                rs_column = col
                break
    
    if not rs_column:
        rs_column = 'rs_no'  # Default fallback
    
    print(f"🔍 Using '{rs_column}' as RS.NO identifier")
    
    # Load returnable plots
    print("📥 Loading returnable plots...")
    returnable_gdf = gpd.read_file(inputs['returnable_plots_shapefile'])
    
    # Ensure same CRS
    if returnable_gdf.crs != result_3.crs:
        print("🔄 Reprojecting returnable plots to match result_3 CRS...")
        returnable_gdf = returnable_gdf.to_crs(result_3.crs)
    
    # Create unified returnable geometry
    print("🔗 Creating unified returnable geometry...")
    returnable_union = returnable_gdf.union_all()
    
    # Remove RS.NO plots that intersect with returnable plots
    print("✂️  Removing RS.NO plots intersecting with returnable plots...")
    intersects_returnable = result_3.geometry.intersects(returnable_union)
    result_3_filtered = result_3[~intersects_returnable].copy()
    
    print(f"🗑️  Removed {len(result_3) - len(result_3_filtered)} RS.NO plots intersecting with returnable plots")
    
    # Filter out category equal to land acquisition
    exclude_category = inputs.get('exclude_category', 'Land Acquisition')
    
    # Find category column
    category_column = None
    for col in result_3_filtered.columns:
        if 'category' in col.lower():
            category_column = col
            break
    
    if category_column:
        print(f"🔍 Filtering out category: '{exclude_category}' from column '{category_column}'...")
        initial_count = len(result_3_filtered)
        result_3_filtered = result_3_filtered[result_3_filtered[category_column] != exclude_category].copy()
        final_count = len(result_3_filtered)
        print(f"🗑️  Removed {initial_count - final_count} RS.NO plots with category '{exclude_category}'")
    else:
        print("⚠️  Warning: Category column not found. Skipping category filtering.")
    
    # Get unique RS.NO values that have unauthorized cultivation
    print("🔍 Identifying RS.NO plots with unauthorized cultivation...")
    unauthorized_rsno_values = set(result_4[rs_column].unique())
    
    # Filter result_3 to only include RS.NO plots with unauthorized cultivation
    result_5 = result_3_filtered[result_3_filtered[rs_column].isin(unauthorized_rsno_values)].copy()
    
    # Generate unauthorized RS.NO UUIDs
    result_5["unauthorized_rsno_uuid"] = [str(uuid.uuid4()) for _ in range(len(result_5))]
    
    print(f"✅ Generated {len(result_5)} unauthorized RS.NO plots")
    
    # Save Result 5
    result_5_path = os.path.join(inputs['output_dir'], "result_5_unauthorized_rsno_plots.shp")
    result_5_geojson = os.path.join(inputs['output_dir'], "result_5_unauthorized_rsno_plots.geojson")
    result_5_csv = os.path.join(inputs['output_dir'], "result_5_unauthorized_rsno_plots.csv")
    
    result_5.to_file(result_5_path)
    result_5.to_file(result_5_geojson, driver="GeoJSON")
    
    # Create CSV version
    result_5_csv_data = result_5.drop(columns=['geometry']).copy()
    result_5_csv_data.to_csv(result_5_csv, index=False)
    
    print(f"💾 Saved Result 5 to:")
    print(f"   - {result_5_path}")
    print(f"   - {result_5_geojson}")
    print(f"   - {result_5_csv}")
    
    return result_5, result_5_path


def generate_summary_report(inputs, result_2, result_3, result_4, result_5):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    summary_data = {
        'Metric': [
            'Total Cultivation Areas with Mosaic Properties',
            'RS.NO Plots with Significant Cultivation (>=15%)',
            'Unauthorized Cultivation Land Polygons',
            'Unauthorized RS.NO Plots',
            'Total Unauthorized Cultivation Area (acres)',
            'Total Unauthorized RS.NO Area (acres)'
        ],
        'Count': [
            len(result_2),
            len(result_3),
            len(result_4),
            len(result_5),
            f"{result_4['cultivation_area_ac'].sum():.2f}" if 'cultivation_area_ac' in result_4.columns else 'N/A',
            f"{result_5['mosaic_area_ac'].sum():.2f}" if 'mosaic_area_ac' in result_5.columns else 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary report
    summary_path = os.path.join(inputs['output_dir'], "processing_summary_report.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("📊 PROCESSING SUMMARY:")
    print(summary_df.to_string(index=False))
    print(f"\n💾 Summary report saved to: {summary_path}")
    
    return summary_df


def main():
    """Main function to orchestrate the entire workflow"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Enhanced geospatial processing script for cultivation land analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
python script.py --predicted_cultivation_shapefile cultivation.shp 
                --capital_city_mosaic_shapefile mosaic.shp 
                --returnable_plots_shapefile returnable.shp 
                --output_dir ./output

python script.py --predicted_cultivation_shapefile cultivation.shp 
                --capital_city_mosaic_shapefile mosaic.shp 
                --returnable_plots_shapefile returnable.shp 
                --output_dir ./output 
                --exclude_category "Land Acquisition" 
                --utm_epsg 32644 
                --cultivation_threshold_percentage 15.0
            """
        )
        
        # Required arguments
        parser.add_argument('--predicted_cultivation_shapefile', required=True,
                            help='Path to predicted cultivation lands shapefile')
        parser.add_argument('--capital_city_mosaic_shapefile', required=True,
                            help='Path to capital city mosaic shapefile')
        parser.add_argument('--returnable_plots_shapefile', required=True,
                            help='Path to returnable plots shapefile')
        parser.add_argument('--output_dir', required=True,
                            help='Output directory path for all generated files')
        
        # Optional arguments
        parser.add_argument('--exclude_category', default='Land Acquisition',
                            help='Category to exclude during filtering (default: "Land Acquisition")')
        parser.add_argument('--utm_epsg', default='32644',
                            help='UTM EPSG code for area calculation (default: 32644 for AP)')
        parser.add_argument('--cultivation_threshold_percentage', type=float, default=15.0,
                            help='Minimum cultivation percentage for RS.NO inclusion (default: 15.0)')
        
        args = parser.parse_args()
        
        # Convert args to dictionary
        inputs = {
            'predicted_cultivation_shapefile': args.predicted_cultivation_shapefile,
            'capital_city_mosaic_shapefile': args.capital_city_mosaic_shapefile,
            'returnable_plots_shapefile': args.returnable_plots_shapefile,
            'output_dir': args.output_dir,
            'exclude_category': args.exclude_category,
            'utm_epsg': args.utm_epsg,
            'cultivation_threshold_percentage': args.cultivation_threshold_percentage
        }
        
        # Print configuration
        print("=" * 70)
        print("ENHANCED GEOSPATIAL DATA PROCESSING SCRIPT")
        print("=" * 70)
        print("📋 Configuration:")
        print(f"   Predicted cultivation shapefile: {inputs['predicted_cultivation_shapefile']}")
        print(f"   Capital city mosaic shapefile: {inputs['capital_city_mosaic_shapefile']}")
        print(f"   Returnable plots shapefile: {inputs['returnable_plots_shapefile']}")
        print(f"   Output directory: {inputs['output_dir']}")
        print(f"   Exclude category: {inputs['exclude_category']}")
        print(f"   UTM EPSG: {inputs['utm_epsg']}")
        print(f"   Cultivation threshold: {inputs['cultivation_threshold_percentage']}%")
        
        # Validate inputs
        validate_inputs(inputs)
        
        # Execute processing steps
        print("\n Starting processing workflow...")
        
        # Result 2: Clip cultivation with mosaic
        result_2, result_2_path = result_2_clip_cultivation_with_mosaic(inputs)
        
        # Result 3: Extract RS.NO polygons with significant cultivation
        result_3, result_3_path = result_3_extract_rsno_polygons(inputs, result_2)
        
        # Result 4: Get unauthorized cultivation lands
        result_4, result_4_path = result_4_unauthorized_cultivation_lands(inputs, result_2)
        
        # Result 5: Get unauthorized RS.NO plots
        result_5, result_5_path = result_5_unauthorized_rsno_plots(inputs, result_3, result_4)
        
        # Generate summary report
        summary_df = generate_summary_report(inputs, result_2, result_3, result_4, result_5)
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 ALL PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 All outputs saved in: {inputs['output_dir']}")
        print("\n📋 Generated Files:")
        print("   Result 2: Cultivation lands with mosaic properties")
        print("   Result 3: RS.NO polygons with significant cultivation")
        print("   Result 4: Unauthorized cultivation land polygons") 
        print("   Result 5: Unauthorized RS.NO plots")
        print("   Summary: Processing summary report")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please check your inputs and try again.")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()