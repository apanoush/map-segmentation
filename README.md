# Segmented Image to Shapefile Converter

This tool converts segmented PNG images (with buildings marked in pink) to shapefiles using geotiff files for projection information.

## SOURCES

<https://github.com/>

## Requirements

Install the required dependencies:

```bash
pip install -e .
# OR
uv sync
```

## Usage

### Basic Usage

```python
from src.segmented_image2shp import segmented_image_to_shp

# Convert segmented PNG to shapefile
gdf = segmented_image_to_shp(
    png_path="path/to/segmented_image.png",
    geotiff_path="path/to/georeferenced_image.tif",
    output_shp_path="output/buildings.shp",
    building_color=(255, 0, 255),  # Pink color (RGB)
    buffer_distance=20  # Pixels
)
```

### Process Geotiffs from Config

```bash
python process_geotiffs.py
```

The script will process both 1973 and 1983 geotiffs defined in `config.toml`.

## Input Requirements

1. **Segmented PNG Image**:
   - Buildings should be marked in pink (RGB: 255, 0, 255)
   - Other areas can be any color
   - The image should be a standard PNG file

2. **Geotiff File**:
   - Must contain proper georeferencing information
   - Should cover the same area as the PNG image
   - Must have projection information embedded

## Output

The script generates a shapefile containing:
- Polygon geometries for each building
- Area and perimeter attributes for each polygon
- Proper projection from the geotiff file

## Algorithm Steps

1. **Load PNG**: Read the segmented image
2. **Identify Buildings**: Find pink pixels (RGB: 255, 0, 255)
3. **Apply Buffer**: Create a buffer around building areas
4. **Distance Mask**: Set pixels beyond buffer to black
5. **Find Contours**: Extract building outlines using skimage
6. **Load Projection**: Read geotransform from geotiff using rasterio
7. **Convert Coordinates**: Transform pixel coordinates to geographic coordinates
8. **Create Shapefile**: Save polygons with projection

## Notes

- The buffer distance (default: 20 pixels) helps connect nearby building pixels
- Small contours (< 3 points) are filtered out
- Only valid polygons with positive area are included in the output
- The script handles common errors and provides informative messages
