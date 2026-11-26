"""Collect camera, weather, and truck restriction CSVs for the CCTV project."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile
from zoneinfo import ZoneInfo


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
BANGKOK = ZoneInfo("Asia/Bangkok")
HOUR = timedelta(hours=1)
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
CAMERA_SOURCE = "https://drive.google.com/drive/folders/1k2-KxXe4wbDmFGcGXI-pK7TdSXjBHTni"
IMAGE_NAME = re.compile(r"^(\d+)_.+_(\d{8})_(\d{6})\.jpg$", re.IGNORECASE)


def download(url):
    request = Request(url, headers={"User-Agent": "DSDE-CEDT-CCTV academic project/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def get_json(url):
    result = json.loads(download(url))
    if isinstance(result, dict) and result.get("error"):
        raise ValueError(f"Source error at {url}: {result.get('reason', result['error'])}")
    return result


def read_cameras(metadata):
    cameras = {}
    with metadata.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            camera_id = row["camera_id"].strip()
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            if not camera_id or camera_id in cameras:
                raise ValueError(f"Invalid or duplicate camera_id: {camera_id!r}")
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError(f"Invalid coordinates for camera {camera_id}")
            detail_en = row["location_detail_en"].strip()
            direction = detail_en if detail_en.lower().startswith(
                ("from ", "toward", "facing ", "the front of ")
            ) else ""
            cameras[camera_id] = {
                "camera_id": camera_id,
                "camera_name": row["location"].strip(),
                "latitude": latitude,
                "longitude": longitude,
                "direction": direction,
                "location_detail": row["location_detail"].strip(),
                "location_detail_en": detail_en,
                "source_url": CAMERA_SOURCE,
            }
    if not cameras:
        raise ValueError(f"No cameras in {metadata}")
    return cameras


def capture_dates(archive, cameras):
    dates = set()
    camera_ids = set()
    with ZipFile(archive) as images:
        for member in images.namelist():
            path = PurePosixPath(member)
            if path.suffix.lower() != ".jpg" or path.parts[0] != "cctv_images":
                continue
            match = IMAGE_NAME.fullmatch(path.name)
            if not match or match[1] != path.parent.name:
                raise ValueError(f"Unexpected CCTV image filename: {member}")
            camera_ids.add(match[1])
            dates.add(date.fromisoformat(match[2]))
    if camera_ids != cameras.keys():
        raise ValueError(f"Camera IDs differ between the ZIP and CSV: {camera_ids ^ cameras.keys()}")
    return min(dates), max(dates)


def weather_rows(cameras, start, end):
    # The provider labels rain by the end of the hour. Fetch the next midnight too.
    first = datetime.combine(start, time.min, BANGKOK)
    stop = datetime.combine(end + timedelta(days=1), time.min, BANGKOK)
    expected_hours = (stop - first) // HOUR
    by_coordinate = defaultdict(list)
    for camera in cameras.values():
        by_coordinate[camera["latitude"], camera["longitude"]].append(camera["camera_id"])

    coordinates = list(by_coordinate)
    for index in range(0, len(coordinates), 20):
        batch = coordinates[index:index + 20]
        parameters = {
            "hourly": "rain",
            "timezone": BANGKOK.key,
            "models": "ecmwf_ifs",
            "start_date": start.isoformat(),
            "end_date": stop.date().isoformat(),
        }
        url = WEATHER_API + "?" + urlencode(parameters | {
            "latitude": ",".join(str(lat) for lat, _ in batch),
            "longitude": ",".join(str(lon) for _, lon in batch),
        })
        response = get_json(url)
        locations = response if isinstance(response, list) else [response]
        for coordinate, location in zip(batch, locations, strict=True):
            source_url = WEATHER_API + "?" + urlencode(parameters | {
                "latitude": coordinate[0], "longitude": coordinate[1],
            })
            if location["utc_offset_seconds"] != 25200 or location["hourly_units"]["rain"] != "mm":
                raise ValueError("Open-Meteo returned an unexpected timezone or rain unit")
            if max(abs(coordinate[0] - location["latitude"]),
                   abs(coordinate[1] - location["longitude"])) > 0.12:
                raise ValueError("Open-Meteo returned a grid point far from the requested location")
            matched_hours = set()
            for label, rain in zip(location["hourly"]["time"], location["hourly"]["rain"], strict=True):
                hour_start = datetime.fromisoformat(label).replace(tzinfo=BANGKOK) - HOUR
                if not first <= hour_start < stop:
                    continue
                if (hour_start - first) % HOUR or hour_start in matched_hours:
                    raise ValueError(f"Unexpected or duplicate weather hour: {label}")
                matched_hours.add(hour_start)
                if rain is not None and (not math.isfinite(rain) or rain < 0):
                    raise ValueError(f"Invalid rain amount at {label}: {rain}")
                for camera_id in by_coordinate[coordinate]:
                    yield {
                        "camera_id": camera_id,
                        "hour_start": hour_start.isoformat(timespec="minutes"),
                        "rain_mm": rain,
                        "grid_latitude": location["latitude"],
                        "grid_longitude": location["longitude"],
                        "weather_model": "ECMWF IFS",
                        "source_url": source_url,
                    }
            if len(matched_hours) != expected_hours:
                raise ValueError(f"Open-Meteo returned {len(matched_hours)} of {expected_hours} hours")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8-sig", newline="", delete=False)
    temporary = Path(output.name)
    try:
        with output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{path}: {len(rows):,} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("all", "cameras", "rain", "rules"), nargs="?", default="all")
    parser.add_argument("--metadata", type=Path, default=ROOT / "cctv_info.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "cctv_images.zip")
    parser.add_argument("--road-extract", type=Path, help="reuse a downloaded Bangkok.osm.geojson.xz")
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "data")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    if bool(args.start) != bool(args.end) or (args.start and args.end < args.start):
        parser.error("provide both --start and --end in chronological order")

    if args.source in ("all", "cameras", "rain"):
        cameras = read_cameras(args.metadata)
    if args.source in ("all", "cameras"):
        from roads import camera_rows

        write_csv(args.output / "cameras.csv", list(camera_rows(cameras, download, args.road_extract)))
    if args.source in ("all", "rain"):
        start, end = (args.start, args.end) if args.start else capture_dates(args.images, cameras)
        rain = sorted(weather_rows(cameras, start, end), key=lambda row: (row["camera_id"], row["hour_start"]))
        write_csv(args.output / "rain_hourly.csv", rain)
    if args.source in ("all", "rules"):
        from rules import restriction_rows

        write_csv(args.output / "truck_rules.csv", list(restriction_rows(download)))


if __name__ == "__main__":
    main()
