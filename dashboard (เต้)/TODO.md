# TODO — dashboard (เต้)

**Criterion:** Component 3, BI Dashboard. Also the demo part of the video.
**Target date:** Thu 3 Dec 2026.

## Status

- [x] Streamlit app (`app.py`): KPIs, hourly chart with ban windows, camera × hour heatmap, camera ranking,
      map, evidence gallery
- [x] Violation time-lapse clip (`make_timelapse.py` → `data_out/truck_violations.mp4`)

## Remaining

- [ ] **Deploy** (e.g. Streamlit Community Cloud) so graders can open it. Needs the data files in the repo and
      relative paths (see `pipeline (เก้อ)/TODO.md`). For the gallery, commit a small set of evidence crops
      instead of the raw images.
- [ ] **Add scraped data**: rain overlay on the hourly chart, official zone on the map, real camera positions
- [ ] Filters: inside/outside zone, weekday/weekend, model-confidence threshold
- [ ] Optional: hour slider on the map (an in-app version of the clip)
- [ ] **Demo script** (2–3 min) for the video, plus screenshots for the slides
- [ ] Re-render the clip after the official zone / camera coordinates arrive
