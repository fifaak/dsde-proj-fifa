"""Find named road candidates near the camera coordinates."""

import json
import lzma
import math
from io import BytesIO


OSM_EXTRACT = "https://download.bbbike.org/osm/bbbike/Bangkok/Bangkok.osm.geojson.xz"
DRIVING_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street"}
MAX_DISTANCE_M = 100
AMBIGUITY_M = 25


def named_segments(extract):
    # Osmium's GeoJSON export puts each feature on its own line.
    segments = []
    with lzma.open(extract, "rt", encoding="utf-8") as source:
        for line in source:
            if '"LineString"' not in line or '"highway"' not in line:
                continue
            feature = json.loads(line.strip().rstrip(","))
            properties = feature["properties"]
            name = properties.get("name:th") or properties.get("name") or properties.get("name:en")
            road_class = properties.get("highway", "")
            if not name or road_class.removesuffix("_link") not in DRIVING_ROADS:
                continue
            path = feature["geometry"]["coordinates"]
            if len(path) < 2:
                continue
            xs, ys = zip(*path, strict=True)
            segments.append((name, road_class, path, (min(xs), min(ys), max(xs), max(ys))))
    if not segments:
        raise ValueError("No named roads found in the Bangkok OSM extract")
    return segments


def line_distance_m(x, y, scale_x, path):
    # The local projection is sufficient for distances of a few meters within Bangkok.
    points = [((lon - x) * scale_x, (lat - y) * 111_320) for lon, lat in path]
    nearest = math.inf
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        projection = min(1, max(0, -(ax * dx + ay * dy) / length_squared)) if length_squared else 0
        nearest = min(nearest, math.hypot(ax + projection * dx, ay + projection * dy))
    return nearest


def nearby_roads(camera, segments):
    x, y = camera["longitude"], camera["latitude"]
    scale_x = 111_320 * math.cos(math.radians(y))
    pad_x = MAX_DISTANCE_M / scale_x
    pad_y = MAX_DISTANCE_M / 111_320
    by_name = {}
    for name, road_class, path, (left, bottom, right, top) in segments:
        if not (left - pad_x <= x <= right + pad_x and bottom - pad_y <= y <= top + pad_y):
            continue
        distance = line_distance_m(x, y, scale_x, path)
        if distance <= MAX_DISTANCE_M and distance < by_name.get(name, (math.inf, None))[0]:
            by_name[name] = (distance, road_class)
    return sorted((distance, name, road_class) for name, (distance, road_class) in by_name.items())


def camera_rows(cameras, download, extract=None):
    segments = named_segments(extract if extract else BytesIO(download(OSM_EXTRACT)))
    for camera in sorted(cameras.values(), key=lambda row: row["camera_id"]):
        nearby = nearby_roads(camera, segments)
        distance, name, road_class = nearby[0] if nearby else (None, "", "")
        second_distance, second_name, second_class = nearby[1] if len(nearby) > 1 else (None, "", "")
        status = "nearby_candidate" if nearby else "no_named_road_within_100m"
        if second_distance is not None and second_distance - distance <= AMBIGUITY_M:
            status = "multiple_nearby_candidates"
        yield camera | {
            "road_name": name,
            "road_class": road_class,
            "road_distance_m": round(distance, 1) if distance is not None else None,
            "alternate_road_name": second_name,
            "alternate_road_class": second_class,
            "alternate_road_distance_m": round(second_distance, 1) if second_distance is not None else None,
            "road_match_status": status,
            "road_source_url": OSM_EXTRACT,
        }
