from shapely.geometry import Point, LineString, Polygon, mapping, shape
from shapely.ops import transform, unary_union
from pyproj import CRS, Transformer
import os
from glob import glob
import requests
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


class GeoJSONManager:
    """Class for managing GeoJSON files, including loading, saving, and accessing features."""

    def __init__(self, file_path=None):
        """
        Initialize the GeoJSONManager.

        :param file_path: Optional path to a GeoJSON file to load.
        """
        self.file_path = file_path
        self.data = None
        if file_path:
            self.load(file_path)

    def load(self, file_path):
        """
        Load GeoJSON data from a file.

        :param file_path: Path to the GeoJSON file.
        :return: Loaded GeoJSON data as a dictionary.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        return self.data

    def save(self, output_path):
        """
        Save GeoJSON data to a file.

        :param output_path: Path to save the GeoJSON file.
        """
        if self.data is None:
            raise ValueError("No GeoJSON data to save.")
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)

    def get_features(self):
        """
        Retrieve all features from the loaded GeoJSON.

        :return: List of features from the GeoJSON.
        """
        if self.data is None or "features" not in self.data:
            raise ValueError("Invalid GeoJSON data.")
        return self.data["features"]


class PolygonCreator:
    """Class for creating geometric shapes like circles and buffers around lines."""

    def __init__(self):
        self.transformer_to_aeqd = None

    def create_circle(self, center, radius, crs="EPSG:4326"):
        """
        Create a circular polygon.

        :param center: Tuple of (longitude, latitude) representing the circle's center.
        :param radius: Radius of the circle in meters.
        :param crs: Coordinate reference system for the input coordinates (default is 'EPSG:4326').
        :return: A Shapely Polygon object representing the circle.
        """
        wgs84 = CRS(crs)
        aeqd_proj_string = (
            f"+proj=aeqd +lat_0={center[1]} +lon_0={center[0]} +datum=WGS84"
        )
        aeqd = CRS.from_proj4(aeqd_proj_string)
        self.transformer_to_aeqd = Transformer.from_crs(wgs84, aeqd, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(aeqd, wgs84, always_xy=True)

        # Transform center to AEQD projection
        center_aeqd = self.transformer_to_aeqd.transform(*center)

        # Create a buffer (circle)
        point = Point(center_aeqd)
        buffer = point.buffer(radius)  # Radius in meters

        # Transform back to WGS84
        buffer_wgs84 = transform(transformer_to_wgs84.transform, buffer)
        return buffer_wgs84

    def create_buffer_from_linestring(self, linestring_coords, buffer_distance, crs="EPSG:4326"):
        """
        Create a buffer polygon around a line.

        :param linestring_coords: List of tuples representing the line's coordinates (longitude, latitude).
        :param buffer_distance: Distance of the buffer around the line in meters.
        :param crs: Coordinate reference system for the input coordinates (default is 'EPSG:4326').
        :return: A Shapely Polygon object representing the buffer.
        """
        center = linestring_coords[0]
        aeqd_proj_string = (
            f"+proj=aeqd +lat_0={center[1]} +lon_0={center[0]} +datum=WGS84"
        )
        aeqd = CRS.from_proj4(aeqd_proj_string)
        self.transformer_to_aeqd = Transformer.from_crs(CRS(crs), aeqd, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(aeqd, CRS(crs), always_xy=True)

        # Transform LineString to AEQD
        line = LineString([self.transformer_to_aeqd.transform(*coord) for coord in linestring_coords])

        # Create a buffer around the LineString
        buffer = line.buffer(buffer_distance)

        # Transform back to WGS84
        buffer_wgs84 = transform(transformer_to_wgs84.transform, buffer)
        return buffer_wgs84


class PolygonMerger:
    """Class for merging polygons."""

    @staticmethod
    def merge_polygons(geojson_features):
        """
        Merge multiple polygons and multipolygons into one.

        :param geojson_features: List of GeoJSON features with 'Polygon' or 'MultiPolygon' geometries.
        :return: A Shapely geometry object representing the merged polygon.
        """
        polygons = [
            shape(feature["geometry"]) for feature in geojson_features
            if feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}
        ]
        if not polygons:
            raise ValueError("No polygons or multipolygons found to merge.")

        return unary_union(polygons)


def make_circle(center, radius, crs="EPSG:4326"):
    """
    Create a circular polygon and return it as a GeoJSON FeatureCollection.

    :param center: Tuple of (longitude, latitude) representing the circle's center.
    :param radius: Radius of the circle in meters.
    :param crs: Coordinate reference system for the input coordinates (default is 'EPSG:4326').
    :return: A dictionary representing the GeoJSON FeatureCollection containing the circle.
    """
    creator = PolygonCreator()
    circle = creator.create_circle(center, radius, crs)
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(circle), "properties": {"name": "Circle"}}
        ]
    }
    manager = GeoJSONManager()
    manager.data = geojson_data
    return manager


def make_buffer(line_geojson_path, buffer_distance, crs="EPSG:4326"):
    """
    Create a buffer polygon around a line specified in a GeoJSON file and return it as a GeoJSON FeatureCollection.

    :param line_geojson_path: Path to the GeoJSON file containing a LineString geometry.
    :param buffer_distance: Distance of the buffer around the line in meters.
    :param crs: Coordinate reference system for the input coordinates (default is 'EPSG:4326').
    :return: A dictionary representing the GeoJSON FeatureCollection containing the buffer.
    """
    manager = GeoJSONManager(line_geojson_path)
    features = manager.get_features()

    # Find the first LineString in the features
    for feature in features:
        if feature["geometry"]["type"] == "LineString":
            line_coords = feature["geometry"]["coordinates"]
            break
    else:
        raise ValueError("No LineString found in the provided GeoJSON file.")

    creator = PolygonCreator()
    buffer_polygon = creator.create_buffer_from_linestring(line_coords, buffer_distance, crs)

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(buffer_polygon), "properties": {"name": "Buffer"}}
        ]
    }
    manager.data = geojson_data
    return manager


def make_merged_polygons(polygons_dir):
    """
    Merge all polygons and multipolygons from GeoJSON files in a directory into a single GeoJSON FeatureCollection.

    :param polygons_dir: Path to the directory containing GeoJSON files with polygons or multipolygons.
    :return: A dictionary representing the GeoJSON FeatureCollection with merged polygons.
    """
    all_features = []
    for file_path in glob(os.path.join(polygons_dir, "*.geojson")):
        manager = GeoJSONManager(file_path)
        all_features.extend(manager.get_features())

    merged_polygon = PolygonMerger.merge_polygons(all_features)

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(merged_polygon), "properties": {"name": "Merged"}}
        ]
    }
    manager = GeoJSONManager()
    manager.data = geojson_data
    return manager


def make_merged_from_file(input_geojson_path):
    """
    Merge all polygons and multipolygons from a single GeoJSON file into one GeoJSON FeatureCollection.

    :param input_geojson_path: Path to the input GeoJSON file containing polygons or multipolygons.
    :return: A dictionary representing the GeoJSON FeatureCollection with merged polygons.
    """
    manager = GeoJSONManager(input_geojson_path)
    features = manager.get_features()

    merged_polygon = PolygonMerger.merge_polygons(features)

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(merged_polygon), "properties": {"name": "Merged"}}
        ]
    }
    manager.data = geojson_data
    return manager


def darken_color(hex_color, factor=0.7):
    """
    Darken a HEX color by reducing brightness.

    :param hex_color: Color in HEX format (e.g., '#FFB3BA').
    :param factor: Brightness reduction factor (0.0 for black, 1.0 for no change).
    :return: HEX color string for the darkened color.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def make_visualization_file(input_geojson_path, output_visualization_path):
    """
    Generate a visualization GeoJSON file with unique styling for each feature.
    - Polygons remain as they are.
    - MultiPolygons are split into individual Polygons with unique names.
    - Includes support for Points and LineStrings.

    :param input_geojson_path: Path to the input GeoJSON file.
    :param output_visualization_path: Path to save the visualization GeoJSON file.
    :return: Processed GeoJSON as a dictionary.
    """
    # color settings
    base_colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E6BAFF", "#FFC4E6"]
    pastel_colors = [
        "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
        "#E6BAFF", "#FFC4E6", "#C4E6FF", "#E6FFC4", "#FFE6C4"
    ]
    stroke_opacity = 0.9
    fill_opacity = 0.4
    stroke_width = 2
    marker_color = "#b51eff"

    manager = GeoJSONManager(input_geojson_path)
    features = manager.get_features()

    if not features:
        raise ValueError(f"No features found in the GeoJSON file: {input_geojson_path}")

    visualization_features = []
    feature_id = 0

    for feature in features:
        geometry = feature.get("geometry")
        properties = feature.get("properties", {})
        if not geometry:
            continue

        pastel_colors = pastel_colors[feature_id % len(pastel_colors)]
        stroke_color = "#e6761b" if feature_id % 2 == 0 else "#793d0e"  # Пример для варьирования

        if geometry["type"] == "Polygon":
            visualization_features.append({
                "type": "Feature",
                "id": feature_id,
                "geometry": geometry,
                "properties": {
                    "description": properties.get("description", f"Feature {feature_id}"),
                    "fill": pastel_colors,
                    "fill-opacity": fill_opacity,
                    "stroke": stroke_color,
                    "stroke-width": stroke_width,
                    "stroke-opacity": stroke_opacity,
                }
            })
            feature_id += 1

        elif geometry["type"] == "MultiPolygon":
            multipolygon = shape(geometry)
            for idx, polygon in enumerate(multipolygon.geoms, start=1):
                visualization_features.append({
                    "type": "Feature",
                    "id": feature_id,
                    "geometry": mapping(polygon),
                    "properties": {
                        "description": f"{properties.get('description', 'MultiPolygon')}_{idx}",
                        "fill": pastel_colors,
                        "fill-opacity": fill_opacity,
                        "stroke": stroke_color,
                        "stroke-width": stroke_width,
                        "stroke-opacity": stroke_opacity,
                    }
                })
                feature_id += 1

        elif geometry["type"] == "LineString":
            visualization_features.append({
                "type": "Feature",
                "id": feature_id,
                "geometry": geometry,
                "properties": {
                    "description": properties.get("description", f"LineString {feature_id}"),
                    "stroke": marker_color,
                    "stroke-width": stroke_width + 2,
                    "stroke-opacity": fill_opacity,
                }
            })
            feature_id += 1

        elif geometry["type"] == "Point":
            visualization_features.append({
                "type": "Feature",
                "id": feature_id,
                "geometry": geometry,
                "properties": {
                    "description": properties.get("description", f"Point {feature_id}"),
                    "iconCaption": properties.get("iconCaption", f"Point {feature_id}"),
                    "marker-color": marker_color,
                }
            })
            feature_id += 1

    visualization_geojson = {
        "type": "FeatureCollection",
        "features": visualization_features
    }

    with open(output_visualization_path, 'w', encoding='utf-8') as file:
        json.dump(visualization_geojson, file, ensure_ascii=False, indent=2)

    logging.info(f"Visualization GeoJSON saved to: {output_visualization_path}")
    return None


def fetch_osm_polygon_by_id(relation_id, name="Region"):
    """
    Fetch a single polygon from OSM by Relation ID.

    :param relation_id: OSM Relation ID.
    :param name: Name of the region for metadata.
    :return: GeoJSON feature with the fetched geometry.
    """
    url = f"http://polygons.openstreetmap.fr/get_geojson.py?id={relation_id}&params=0"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            geo_json = json.loads(response.text)
            geometry = geo_json.get("geometries", [geo_json])[0]

            return {
                'type': 'Feature',
                'geometry': geometry,
                'properties': {
                    'description': name
                }
            }
        except json.JSONDecodeError:
            logging.error(f"Failed to parse GeoJSON for Relation ID {relation_id}")
            return None
    else:
        logging.error(f"Failed to fetch polygon for Relation ID {relation_id} (HTTP {response.status_code})")
        return None


def fetch_osm_polygons_from_excel(file_path, output_directory):
    """
    Fetch polygons from OSM based on an Excel file.

    :param file_path: Path to the Excel file containing 'relation_id' and 'name' columns.
    :param output_directory: Directory where individual GeoJSON files will be saved.
    :return: GeoJSON dictionary containing all fetched polygons.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df = pd.read_excel(file_path)

    if 'relation_id' not in df.columns or 'name' not in df.columns:
        raise ValueError("The Excel file must contain 'relation_id' and 'name' columns.")

    geojson_data = {
        'type': 'FeatureCollection',
        'features': []
    }

    for _, row in df.iterrows():
        relation_id = row.get('relation_id')
        name = row.get('name', 'Unknown Region')

        if not pd.isna(relation_id):
            feature = fetch_osm_polygon_by_id(relation_id=relation_id, name=name)
            if feature:
                geojson_data['features'].append(feature)

                feature_geojson = {
                    'type': 'FeatureCollection',
                    'features': [feature]
                }
                output_file_path = os.path.join(output_directory, f"{name}.geojson")
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(feature_geojson, file, ensure_ascii=False, indent=2)
                logging.info(f"Saved polygon: {output_file_path}")
        else:
            logging.warning(f"Skipping row with missing relation_id: {row}")

    return geojson_data


def merge_all_polygons_into_one(directory):
    """
    Recursively merges all polygons and multipolygons from GeoJSON files in a directory into a single polygon.

    :param directory: Path to the directory containing GeoJSON files.
    :return: GeoJSON dictionary with the merged polygon.
    """
    all_polygons = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".geojson"):
                file_path = os.path.join(root, file)
                logging.info(f"Processing: {file_path}")

                manager = GeoJSONManager(file_path)
                features = manager.get_features()

                for feature in features:
                    geometry = feature.get("geometry")
                    if geometry and geometry["type"] in {"Polygon", "MultiPolygon"}:
                        all_polygons.append(shape(geometry))

    if not all_polygons:
        raise ValueError("No polygons or multipolygons found in the provided directory.")

    merged_polygon = unary_union(all_polygons)

    merged_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(merged_polygon),
                "properties": {"description": "Merged Polygon"}
            }
        ]
    }

    return merged_geojson


def merge_features_from_directory(directory):
    """
    Combine all features from GeoJSON files in a directory into a single FeatureCollection.

    :param directory: Path to the directory containing GeoJSON files.
    :return: GeoJSON dictionary with all features collected from the directory.
    """
    all_features = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".geojson"):
                file_path = os.path.join(root, file)
                logging.info(f"Processing: {file_path}")

                manager = GeoJSONManager(file_path)
                features = manager.get_features()

                all_features.extend(features)

    if not all_features:
        raise ValueError("No features found in the provided directory.")

    merged_geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }

    return merged_geojson


def merge_geojson_directory(input_directory, output_path):
    """
    Collect all features from GeoJSON files in a directory, merge them into one FeatureCollection,
    and save the result to a file.

    :param input_directory: Path to the directory containing GeoJSON files.
    :param output_path: Path to save the merged GeoJSON file.
    """
    merged_geojson = merge_features_from_directory(input_directory)

    manager = GeoJSONManager()
    manager.data = merged_geojson
    manager.save(output_path)

    print(f"Merged GeoJSON saved to: {output_path}")


def collect_polygons_into_one_file(directory):
    """
    Recursively collects all polygons and multipolygons from GeoJSON files in a directory into one GeoJSON file.

    :param directory: Path to the directory containing GeoJSON files.
    :return: GeoJSON dictionary with all polygons collected as separate features.
    """
    all_features = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".geojson"):
                file_path = os.path.join(root, file)
                logging.info(f"Processing: {file_path}")

                manager = GeoJSONManager(file_path)
                features = manager.get_features()

                for feature in features:
                    geometry = feature.get("geometry")
                    if geometry and geometry["type"] in {"Polygon", "MultiPolygon"}:
                        all_features.append({
                            "type": "Feature",
                            "geometry": geometry,
                            "properties": feature.get("properties", {})
                        })

    if not all_features:
        raise ValueError("No polygons or multipolygons found in the provided directory.")

    collected_geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }

    return collected_geojson


def process_excel_to_geojson(excel_path, output_directory):
    """
    Process an Excel file with OSM relation IDs into GeoJSON and save individual polygons.

    :param excel_path: Path to the Excel file.
    :param output_directory: Directory to save individual GeoJSON files for each polygon.
    """
    fetch_osm_polygons_from_excel(excel_path, output_directory)
    logging.info(f"Individual polygons saved to: {output_directory}")


def merge_directory_to_geojson(input_directory, output_geojson_path):
    """
    Simplified function to merge all GeoJSON files in a directory into a single GeoJSON.

    :param input_directory: Directory containing input GeoJSON files.
    :param output_geojson_path: Path to the output merged GeoJSON file.
    """
    merged_geojson = merge_all_polygons_into_one(input_directory)
    manager = GeoJSONManager()
    manager.data = merged_geojson
    manager.save(output_geojson_path)
    logging.info(f"Merged GeoJSON saved to {output_geojson_path}")


def create_circle_geojson(center, radius, output_path, name="Circle"):
    """
    Generate a GeoJSON file for a circle.

    :param center: Tuple of (longitude, latitude) representing the circle's center.
    :param radius: Radius of the circle in meters.
    :param output_path: Path to save the resulting GeoJSON file.
    :param name: Name of the circle for metadata.
    """
    creator = PolygonCreator()
    circle = creator.create_circle(center=center, radius=radius)

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(circle),
                "properties": {"name": name}
            }
        ]
    }

    manager = GeoJSONManager()
    manager.data = geojson_data
    manager.save(output_path)
    logging.info(f"GeoJSON with circle saved: {output_path}")


def create_buffer_geojson(line_coords, buffer_distance, output_path, name="Buffer"):
    """
    Create a buffer polygon around a line and save it as a GeoJSON file.

    :param line_coords: List of tuples representing the line's coordinates (longitude, latitude).
    :param buffer_distance: Distance of the buffer around the line in meters.
    :param output_path: Path to save the resulting GeoJSON file.
    :param name: Name of the buffer for metadata.
    """
    creator = PolygonCreator()
    buffer_polygon = creator.create_buffer_from_linestring(line_coords, buffer_distance)

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(buffer_polygon),
                "properties": {"name": name}
            }
        ]
    }

    manager = GeoJSONManager()
    manager.data = geojson_data
    manager.save(output_path)
    logging.info(f"GeoJSON with buffer saved: {output_path}")


def generate_buffers_from_directory(input_directory, output_directory, buffer_distance=500):
    """
    Recursively processes a directory to find all GeoJSON files with LineString geometries,
    creates buffers around them, and saves the buffers in a new directory.

    :param input_directory: Directory containing GeoJSON files.
    :param output_directory: Directory to save buffer GeoJSON files.
    :param buffer_distance: Buffer distance in meters (default is 500).
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Рекурсивный обход директории
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".geojson"):
                input_file_path = os.path.join(root, file)
                manager = GeoJSONManager(input_file_path)

                # Загружаем GeoJSON и проверяем наличие LineString
                features = manager.get_features()
                for i, feature in enumerate(features):
                    geometry = feature.get("geometry")
                    if geometry and geometry["type"] == "LineString":
                        line_coords = geometry["coordinates"]

                        # Создаём имя выходного файла
                        output_file_name = f"{os.path.splitext(file)[0]}_buffer_{i + 1}.geojson"
                        output_file_path = os.path.join(output_directory, output_file_name)

                        # Генерируем буфер и сохраняем
                        create_buffer_geojson(line_coords=line_coords, buffer_distance=buffer_distance,
                                              output_path=output_file_path)
                        logging.info(f"Buffer saved: {output_file_path}")


def create_polygon_from_osm(relation_id, output_path, name="Region"):
    """
    Fetch a polygon from OSM by Relation ID and save it as a GeoJSON file.

    :param relation_id: The OSM Relation ID to fetch.
    :param output_path: Path to save the resulting GeoJSON file.
    :param name: Name of the region for metadata.
    """
    feature = fetch_osm_polygon_by_id(relation_id=relation_id, name=name)
    if not feature:
        logging.error(f"Failed to fetch polygon for Relation ID: {relation_id}")
        return

    geojson_data = {
        "type": "FeatureCollection",
        "features": [feature]
    }

    manager = GeoJSONManager()
    manager.data = geojson_data
    manager.save(output_path)

    logging.info(f"GeoJSON file created: {output_path}")