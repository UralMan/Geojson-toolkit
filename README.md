# Geojson toolkit

Repository for work with `.geojson` files.

## Features
- Generate circle polygons and buffers
- Copy polygons from OSM
- Combine polygons into one object
- Create `.geojson` for visualizing polygons

## Examples

### Generate circle polygon

```python
from geojson_toolkit import create_circle_geojson

center = (37.6173, 55.7558)  # latitude and longitude
radius = 1000                # radius in meters

create_circle_geojson(center=center, radius=radius, output_path="Circle.geojson")
```

### Generate buffer around line

```python
from geojson_toolkit import create_buffer_geojson

line_coords = [(37.6173, 55.7558), (37.6300, 55.7590)]  # list <lat,long>
buffer_distance = 500                                   # buffer in meters

create_buffer_geojson(line_coords=line_coords,
                      buffer_distance=buffer_distance,
                      output_path="Buffer.geojson")
```

### Combine polygons into one object

```python
from geojson_toolkit import merge_geojson_directory

input_directory = "Input_directory"
output_path = "Output_directory/example_polygon.geojson"

merge_geojson_directory(input_directory, output_path)
```

### Copy polygons from OSM

```python
from geojson_toolkit import create_polygon_from_osm

relation_id = 1275551
output_path = "example_polygon.geojson"
territory_name = "example"

create_polygon_from_osm(relation_id=relation_id, output_path=output_path, name=territory_name)
```

### Copy polygons from OSM

```python
from geojson_toolkit import process_excel_to_geojson

process_excel_to_geojson("relations_id.xlsx", "Example_directory")
```

### Create `.geojson` for visualizing polygons

```python
from geojson_toolkit import make_visualization_file

make_visualization_file("example_polygon.geojson", "visualization_file.geojson")
```






