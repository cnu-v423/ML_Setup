def predict(self, image_path, output_path, batch_size=8):
    """Run ensemble prediction with all models - Memory optimized version"""
    
    # Read image
    with rasterio.open(image_path) as src:
        print(self.channels)
        image = src.read(list(range(1, 4)))
        transform = src.transform
        crs = src.crs
        if crs is None:
            crs = rasterio.crs.CRS.from_epsg(4326)
        image = np.moveaxis(image, 0, -1)
    
    # Create tiles first
    tiles, positions, original_shape, (n_h, n_w) = self.create_tiles_new(image)
    
    # Clear the original image from memory
    del image
    import gc
    gc.collect()
    
    all_predictions = {}
    
    # Process each model separately to avoid memory accumulation
    for name, model in self.models.items():
        print(f"\nRunning prediction for {name}...")
        
        if name == "building":
            print("Processing building model...")
        elif name == "vegetation":
            print("Processing vegetation model...")
            continue  # Skip vegetation for now
        else:
            continue  # Skip other models
        
        # Process tiles in smaller batches to manage memory
        predictions = []
        total_tiles = len(tiles)
        
        # Process in chunks to avoid memory issues
        for chunk_start in range(0, total_tiles, batch_size):
            chunk_end = min(chunk_start + batch_size, total_tiles)
            chunk_tiles = tiles[chunk_start:chunk_end]
            
            print(f"Processing chunk {chunk_start//batch_size + 1}/{(total_tiles + batch_size - 1)//batch_size}")
            
            # Process tiles in this chunk
            processed_chunk_tiles = []
            for tile in chunk_tiles:
                try:
                    # Apply RGB Normalization and preprocessing
                    rgb_tile = self.normalize_tile(tile)
                    processed_tile = self.preprocess_rgb_image(rgb_tile)
                    processed_chunk_tiles.append(processed_tile)
                except Exception as e:
                    print(f"Error processing tile for rgb: {str(e)}")
                    # Use fallback processing or skip this tile
                    continue
            
            # Convert to numpy array for this chunk only
            if processed_chunk_tiles:
                chunk_array = np.array(processed_chunk_tiles)
                
                # Process each tile in the chunk
                for i, tile in enumerate(chunk_array):
                    try:
                        # Apply augmentations
                        augmented_tiles = self.apply_augmentation(tile)
                        aug_predictions = []
                        
                        # Process augmented tiles in smaller sub-batches
                        aug_batch_size = min(4, len(augmented_tiles))
                        for aug_start in range(0, len(augmented_tiles), aug_batch_size):
                            aug_end = min(aug_start + aug_batch_size, len(augmented_tiles))
                            aug_batch = augmented_tiles[aug_start:aug_end]
                            
                            # Get predictions for this augmentation batch
                            batch_preds = model.predict(aug_batch, verbose=0)
                            aug_predictions.extend(batch_preds)
                        
                        # Merge augmented predictions
                        merged_pred = self.merge_augmented_predictions(aug_predictions)
                        predictions.append(merged_pred)
                        
                        # Clear augmentation data from memory
                        del augmented_tiles, aug_predictions
                        
                    except Exception as e:
                        print(f"Error in prediction for tile {chunk_start + i}: {str(e)}")
                        continue
                
                # Clear chunk data from memory
                del chunk_array, processed_chunk_tiles
                gc.collect()
            
            # Optional: Add a small delay to help with memory management
            import time
            time.sleep(0.1)
        
        # Merge all predictions for this model
        print(f"Merging {len(predictions)} predictions for {name}")
        merged = self.merge_predictions(predictions, positions, original_shape)
        all_predictions[name] = merged[:,:,0]  # Remove channel dimension
        
        # Clear predictions from memory before saving
        del predictions
        gc.collect()
        
        # Save individual model prediction
        individual_output = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.splitext(os.path.basename(output_path))[0]}_{name}.tif"
        )
        
        self._save_prediction(individual_output, merged[:,:,0], original_shape, crs, transform, name)
        self._save_probability_map(individual_output.replace('.tif', '_prob.tif'), 
                                 merged[:,:,0], original_shape, crs, transform, name)
        
        print(f"Saved {name} predictions")
        
        # Clear merged data after saving
        del merged
        gc.collect()
    
    # Clear tiles from memory before final processing
    del tiles
    gc.collect()
    
    # Resolve conflicts and create final prediction
    print("\nResolving prediction conflicts...")
    final_mask = self.resolve_conflicts(all_predictions)
    
    final_output = os.path.join(
        os.path.dirname(output_path),
        f"{os.path.splitext(os.path.basename(output_path))[0]}_multiclass.tif"
    )
    
    # Save final prediction
    self._save_final_prediction(final_output, final_mask, original_shape, crs, transform)
    
    # Calculate statistics
    stats = self._calculate_stats(final_mask)
    
    # Final cleanup
    del all_predictions, final_mask
    gc.collect()
    
    return stats

def _save_prediction(self, output_path, prediction, original_shape, crs, transform, name):
    """Helper method to save individual predictions"""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=original_shape[0],
        width=original_shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        binary_mask = (prediction > self.thresholds[name]).astype(np.uint8)
        dst.write(binary_mask, 1)
        dst.update_tags(
            MODEL_NAME=name,
            THRESHOLD=str(self.thresholds[name]),
            CLASS_NAME=name.upper()
        )

def _save_probability_map(self, output_path, prediction, original_shape, crs, transform, name):
    """Helper method to save probability maps"""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=original_shape[0],
        width=original_shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(prediction.astype(np.float32), 1)
        dst.update_tags(
            MODEL_NAME=name,
            CONTENT_TYPE="PROBABILITY_MAP"
        )

def _save_final_prediction(self, output_path, final_mask, original_shape, crs, transform):
    """Helper method to save final multiclass prediction"""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=original_shape[0],
        width=original_shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(final_mask, 1)
        dst.update_tags(
            CLASS_VALUES="0,1,2,3",
            CLASS_NAMES="Background,Building,Vegetation,Water"
        )

def _calculate_stats(self, final_mask):
    """Helper method to calculate class statistics"""
    total_pixels = final_mask.size
    stats = {
        'background_percentage': np.sum(final_mask == self.CLASS_BACKGROUND) / total_pixels * 100,
        'building_percentage': np.sum(final_mask == self.CLASS_BUILDING) / total_pixels * 100,
        'vegetation_percentage': np.sum(final_mask == self.CLASS_VEGETATION) / total_pixels * 100,
        'water_percentage': np.sum(final_mask == self.CLASS_WATER) / total_pixels * 100
    }
    return stats