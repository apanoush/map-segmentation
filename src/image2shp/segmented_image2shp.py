"""
Vectorization using rasterio instead of GDAL:

    load the png => apply a buffer around the building class pixel (pink) => set to black all the pixels further than ca. 20 pixels from these surfaces

    contours (skimage) => retrieve the building polygons
    load the geotif projection (rasterio) => apply the projection to the vectors => save the projected polygons as shapefile
"""

"""
Algorithm steps implemented:
  1. Load the PNG image
  2. Identify building pixels (pink/magenta)
  3. Apply buffer around building pixels
  4. Create distance mask
  5. Find contours using skimage
  6. Load geotiff projection with rasterio
"""

import numpy as np
import cv2
from skimage import measure
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
import warnings

warnings.filterwarnings("ignore")


def segmented_image_to_shp(
    png_path,
    geotiff_path,
    output_shp_path,
    building_color=(255, 0, 255),
    buffer_distance=20,
):
    """
    Convert segmented PNG image to shapefile using geotiff for projection (rasterio version).

    Args:
        png_path: Path to segmented PNG image
        geotiff_path: Path to geotiff file for projection information
        output_shp_path: Path for output shapefile
        building_color: RGB color of building pixels (default: pink/magenta)
        buffer_distance: Distance in pixels for buffer around buildings
    """

    # Step 1: Load the PNG image
    print(f"Loading PNG image from {png_path}")
    img = cv2.imread(png_path)
    if img is None:
        raise ValueError(f"Could not load image from {png_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 2: Identify building pixels (pink/magenta)
    print("Identifying building pixels...")
    building_mask = np.all(img_rgb == building_color, axis=2)

    if not np.any(building_mask):
        raise ValueError("No building pixels found with the specified color")

    # Step 3: Apply buffer around building pixels
    print(f"Applying buffer of {buffer_distance} pixels around buildings...")
    kernel_size = 2 * buffer_distance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate the building mask to create buffer
    building_mask_uint8 = building_mask.astype(np.uint8) * 255
    buffered_mask = cv2.dilate(building_mask_uint8, kernel, iterations=1)
    # https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html
    # Step 4: Set to black all pixels further than buffer distance
    print("Creating distance mask...")
    distance_mask = buffered_mask > 0

    # Create output image with only buffered areas
    result_img = np.zeros_like(img_rgb)
    result_img[distance_mask] = img_rgb[distance_mask]

    # Step 5: Find contours using skimage
    print("Finding contours...")
    # Convert to binary mask for contour finding
    binary_mask = distance_mask.astype(np.uint8) * 255

    # Find contours
    contours = measure.find_contours(binary_mask, level=128)

    if not contours:
        raise ValueError("No contours found in the image")

    print(f"Found {len(contours)} contours")

    # Step 6: Load geotiff projection with rasterio
    print(f"Loading geotiff projection from {geotiff_path}")
    try:
        with rasterio.open(geotiff_path) as src:
            transform = src.transform
            crs = src.crs

            if crs is None:
                raise ValueError("Geotiff does not have CRS information")

            # Get CRS as string for geopandas
            crs_str = crs.to_string()

            # Step 7: Convert contours to polygons with projection
            print("Converting contours to projected polygons...")
            polygons = []

            for contour in contours:
                if len(contour) < 3:
                    continue  # Skip contours with less than 3 points

                # Convert pixel coordinates to geographic coordinates
                geo_coords = []
                # contour contains (row, column) pairs
                for row, col in contour:
                    # Transform (col, row) to geographic coordinates
                    # Using rasterio transform: (x, y) = transform * (col, row)
                    x_geo, y_geo = transform * (col, row)
                    geo_coords.append((x_geo, y_geo))

                # Create polygon (ensure it's closed)
                if len(geo_coords) >= 3:
                    poly = Polygon(geo_coords)
                    if poly.is_valid and poly.area > 0:
                        polygons.append(poly)
    except Exception as e:
        raise ValueError(f"Could not open or read geotiff file {geotiff_path}: {e}")

    if not polygons:
        raise ValueError("No valid polygons created from contours")

    print(f"Created {len(polygons)} valid polygons")

    # Step 8: Create GeoDataFrame and save as shapefile
    print(f"Saving shapefile to {output_shp_path}")
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs_str)

    # Add some basic attributes
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length

    # Save to shapefile
    gdf.to_file(output_shp_path)

    print(f"Successfully saved {len(polygons)} polygons to {output_shp_path}")

    return gdf


def main():
    """
    Example usage of the function.
    Modify these paths according to your data.
    """
    # Example paths - modify these according to your data
    png_path = "path/to/your/segmented_image.png"
    geotiff_path = "path/to/your/georeferenced_image.tif"
    output_shp_path = "path/to/your/output/buildings.shp"

    try:
        gdf = segmented_image_to_shp(png_path, geotiff_path, output_shp_path)
        print("Conversion completed successfully!")
        print(f"Total buildings: {len(gdf)}")
        print(f"Total area: {gdf['area'].sum():.2f} square units")
    except Exception as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    main()
