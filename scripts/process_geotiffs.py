#!/usr/bin/env python3
"""
Process geotiffs from config file: segment them and convert to shapefiles.
"""

import sys, os

from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, ".")

print(os.listdir("src"))

from src.image2shp.all_pipeline import process_geotiff
from src.image2shp.config import load_config



def main():
    """Main function to process both geotiffs from config."""
    print("Image2SHP Processing Pipeline")
    print("=" * 60)

    # Load configuration
    config_path = Path("config.docker.toml")
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please create config.toml with the required paths.")
        return 1

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1

    # Get paths from config
    data_paths = config.paths.data
    model_paths = config.paths.model
    result_paths = config.paths.results
    segmentation_config = config.segmentation

    print(f"Images directory: {data_paths.images_dir}")
    print(f"1973 geotiff: {data_paths.georeferenced_tiff_1973_path}")
    print(f"1983 geotiff: {data_paths.georeferenced_tiff_1983_path}")
    print(f"Model config: {model_paths.config_path}")
    print(f"Model checkpoint: {model_paths.checkpoint_path}")
    print(f"Building class index: {segmentation_config.building_class_idx}")
    print(f"Buffer distance: {segmentation_config.buffer_distance}")
    print(f"Segmented PNG output directory: {result_paths.segmented_png_results_dir}")
    print(f"Shapefile output directory: {result_paths.shp_results_dir}")

    # Process both geotiffs
    success_1973 = process_geotiff(
        geotiff_path=data_paths.georeferenced_tiff_1973_path,
        output_png_dir=result_paths.segmented_png_results_dir,
        output_shp_dir=result_paths.shp_results_dir,
        model_config_path=model_paths.config_path,
        model_checkpoint_path=model_paths.checkpoint_path,
        building_class_idx=segmentation_config.building_class_idx,
        buffer_distance=segmentation_config.buffer_distance,
        year="1973",
    )

    success_1983 = process_geotiff(
        geotiff_path=data_paths.georeferenced_tiff_1983_path,
        output_png_dir=result_paths.segmented_png_results_dir,
        output_shp_dir=result_paths.shp_results_dir,
        model_config_path=model_paths.config_path,
        model_checkpoint_path=model_paths.checkpoint_path,
        building_class_idx=segmentation_config.building_class_idx,
        buffer_distance=segmentation_config.buffer_distance,
        year="1983",
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("Processing Summary")
    print(f"{'=' * 60}")
    print(f"1973: {'✓ SUCCESS' if success_1973 else '✗ FAILED'}")
    print(f"1983: {'✓ SUCCESS' if success_1983 else '✗ FAILED'}")

    if success_1973 and success_1983:
        print(f"\nAll processing completed successfully!")
        return 0
    else:
        print(f"\nSome processing failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
