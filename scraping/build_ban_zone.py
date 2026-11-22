"""Build an APPROXIMATE truck-ban zone polygon -> config/ban_zone.geojson, and tag each camera inside/outside.

The official ban area is defined by a police announcement as a list of roads in inner Bangkok; no machine-readable
map was available here. As a stand-in, the zone is the area enclosed by the Ratchadaphisek inner ring road
(ถนนวงแหวนรัชดาภิเษก), traced through junctions geocoded with OpenStreetMap Nominatim, buffered 400 m so cameras
on the ring road itself count as inside. Replace config/ban_zone.geojson with the official polygon when available.

Usage: python scraping/build_ban_zone.py   (uses data_out/geocode_cache.json, 1 req/s to Nominatim for new names)
Output: config/ban_zone.geojson, data_out/cameras_geo.csv gains column in_zone
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import transform  # deprecated alias, still fine for this one-off

sys.path.insert(0, str(Path(__file__).parent))
import geocode_cameras as gc  # noqa: E402

# clockwise around the inner ring, starting at Rama 7 bridge (north-west river crossing)
RING = [
    ["สะพานพระราม 7"], ["แยกวงศ์สว่าง"], ["แยกประชานุกูล"], ["แยกรัชโยธิน"], ["ศาลอาญา"],
    ["แยกรัชดา-ลาดพร้าว"], ["แยกห้วยขวาง"], ["ศูนย์วัฒนธรรมแห่งประเทศไทย"], ["เซ็นทรัล พระราม 9"],
    ["แยกอโศก-เพชรบุรี", "สถานีเพชรบุรี"], ["แยกอโศกมนตรี", "สถานีสุขุมวิท"],
    ["ศูนย์การประชุมแห่งชาติสิริกิติ์"], ["แยก ณ ระนอง"], ["ถนนนราธิวาสราชนครินทร์"], ["สะพานกรุงเทพ"],
    ["แยกบุคคโล"], ["แยกรัชดา-ราชพฤกษ์"], ["สถานีท่าพระ", "Tha Phra Intersection"], ["แยกไฟฉาย"], ["แยกบางพลัด", "สะพานกรุงธน"],
]
BUFFER_M = 400


def to_merc(x, y, z=None):
    return np.radians(x) * 6378137.0, np.log(np.tan(np.pi / 4 + np.radians(y) / 2)) * 6378137.0


def to_ll(x, y, z=None):
    return np.degrees(np.asarray(x) / 6378137.0), np.degrees(2 * np.arctan(np.exp(np.asarray(y) / 6378137.0)) - np.pi / 2)


def main():
    cache = json.loads(gc.CACHE.read_text()) if gc.CACHE.exists() else {}
    pts = []
    for qs in RING:
        hit = next((h for q in qs if (h := gc.search(q, cache))), None)
        if hit:
            pts.append((float(hit["lon"]), float(hit["lat"]), qs[0]))
        else:
            print("not found:", qs)
    for lon, lat, name in pts:
        print(f"{lat:.4f} {lon:.4f} {name}")
    poly = transform(to_merc, Polygon([(lon, lat) for lon, lat, _ in pts])).buffer(BUFFER_M)
    poly_ll = transform(to_ll, poly)
    gj = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(poly_ll),
        "properties": {"name": "เขตห้ามรถบรรทุก (ประมาณ: ภายในวงแหวนรัชดาภิเษก)", "approximate": True,
                       "buffer_m": BUFFER_M, "waypoints": [n for *_, n in pts]}}]}
    Path("config/ban_zone.geojson").write_text(json.dumps(gj, ensure_ascii=False))
    geo = pd.read_csv("data_out/cameras_geo.csv")
    geo["in_zone"] = [poly_ll.contains(Point(lo, la)) for lo, la in zip(geo.lon, geo.lat)]
    geo.to_csv("data_out/cameras_geo.csv", index=False)
    print("cameras in zone:", int(geo.in_zone.sum()), "outside:", geo.loc[~geo.in_zone, "camera_id"].tolist())


if __name__ == "__main__":
    main()
