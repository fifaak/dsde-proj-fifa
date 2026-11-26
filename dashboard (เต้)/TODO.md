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

## Remaining

- [ ] Validate rainfall, the official zone and real camera positions after scraping and pipeline integration.
      Check that the map, ranking, and clip use the same integrated data.
- [ ] Re-render the clip once the official zone and camera coordinates arrive.
- [ ] Refresh screenshots for the final slides after the data update.
- [ ] Deploy the dashboard and verify the link from a viewer session. The entry point is `dashboard (เต้)/app.py`;
      its dependencies are in `dashboard (เต้)/requirements.txt`.
