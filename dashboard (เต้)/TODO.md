# TODO — dashboard (เต้)

**Criterion:** Component 3, BI Dashboard. Also the demo part of the video.
**Target date:** Thu 3 Dec 2026.

## Run locally

Run from the dashboard folder to load its bundled theme.

```bash
cd "dashboard (เต้)"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Status

- [x] Streamlit app with KPIs, a zone map and 10-minute time slider, hourly profile, camera × hour heatmap, and rankings
- [x] Filters for date, camera, day type, zone, and model confidence
- [x] Gallery uses the evidence pack already in the repository, with local CCTV crops when available
- [x] Hourly chart accepts `rain_mm` in the aggregated table or `data_out/weather_hourly.csv`
- [x] Demo script for the video (`demo.md`)
- [x] Time-lapse clip (`make_timelapse.py` → `data_out/truck_violations.mp4`)

## Integrated data

- [x] Verified the pipeline outputs for all 100 course camera locations and 10,655 camera-hours with rainfall.
- [x] Updated the dashboard, demo and slide screenshots to use the integrated data.
- [x] Re-rendered the time-lapse from the integrated sightings and camera coordinates.
      It shows the windows and 10-minute slots with sufficient camera coverage.
- Local screenshots are in `data_out/dashboard_map.png`, `data_out/dashboard_rain.png` and
  `data_out/dashboard_gallery.png`.

The map uses the approximate study boundary. Its 3,028 sightings are counted using the project's weekday
6-wheel reference windows. The rules file also records the sourced 10-wheel area rule and the exemption
applied by the pipeline.

## Remaining

- [x] Publish the tested updates to GitHub for deployment.
- [ ] Deploy from `fifaak/dsde-proj-fifa`, branch `main`, entry point `dashboard (เต้)/app.py`, then verify the
      public URL in a viewer session. Streamlit Community Cloud requires a repository admin to deploy the app.
