
from pathlib import Path
from src.image2shp.segment_image import segment_image_to_png
from src.image2shp.segmented_image2shp import segmented_image_to_shp


def process_geotiff(
    geotiff_path: Path,
    output_png_dir: Path,
    output_shp_dir: Path,
    model_config_path: Path,
    model_checkpoint_path: Path,
    building_class_idx: int,
    buffer_distance: int,
    year: str,
):
    """
    Process a single geotiff: segment it and convert to shapefile.

    Args:
        geotiff_path: Path to geotiff file
        output_png_dir: Directory to save segmented PNG
        output_shp_dir: Directory to save shapefile
        model_config_path: Path to model config file
        model_checkpoint_path: Path to model checkpoint file
        building_class_idx: Class index for buildings in segmentation output
        buffer_distance: Buffer distance in pixels for building contours
        year: Year identifier for naming (e.g., "1973", "1983")
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {year} geotiff: {geotiff_path}")
    print(f"{'=' * 60}")

    # Check if geotiff exists
    if not geotiff_path.exists():
        print(f"ERROR: Geotiff file not found: {geotiff_path}")
        return False

    # Create output directories
    output_png_dir.mkdir(parents=True, exist_ok=True)
    output_shp_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    geotiff_stem = geotiff_path.stem
    png_filename = f"{geotiff_stem}_segmented.png"
    shp_filename = f"{geotiff_stem}_buildings.shp"

    png_path = output_png_dir / png_filename
    shp_path = output_shp_dir / shp_filename

    try:
        # Step 1: Segment geotiff to PNG
        print(f"\n1. Segmenting geotiff to PNG...")
        print(f"   Input: {geotiff_path}")
        print(f"   Output: {png_path}")

        segment_image_to_png(
            img_path=str(geotiff_path),
            output_png_path=str(png_path),
            config_path=model_config_path,
            checkpoint_path=model_checkpoint_path,
            building_class_idx=building_class_idx,
            building_color=(255, 0, 255),  # Pink
        )

        # Step 2: Convert segmented PNG to shapefile
        print(f"\n2. Converting segmented PNG to shapefile...")
        print(f"   Input PNG: {png_path}")
        print(f"   Geotiff for projection: {geotiff_path}")
        print(f"   Output shapefile: {shp_path}")

        gdf = segmented_image_to_shp(
            png_path=str(png_path),
            geotiff_path=str(geotiff_path),
            output_shp_path=str(shp_path),
            building_color=(255, 0, 255),
            buffer_distance=buffer_distance,
        )

        print(f"\n✓ Successfully processed {year} geotiff!")
        print(f"  - Created segmented PNG: {png_path}")
        print(f"  - Created shapefile with {len(gdf)} buildings: {shp_path}")

        return True

    except Exception as e:
        print(f"\n✗ Error processing {year} geotiff:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

