# TODO — analysis (พัท)

**Criterion:** Component 2, Data Science: patterns, trends and insights from the prepared data.
**Target date:** Thu 3 Dec 2026 (feeds insights + slides).

## Status

- [x] EDA of the 61,319 images (`eda_cctv.ipynb` / `.html`): coverage, outages, image quality, lighting

## Remaining

- [ ] **Truck analysis notebook** (`trucks.ipynb`), from `agg_camera_hour.csv` / `heavy_sightings.parquet`:
  - [ ] heavy trucks per frame by hour, weekday vs weekend, inside vs outside the zone
  - [ ] ban effect: weekday ban hours vs 10–14h, compared with the weekend, with a bootstrap CI
        (current result 0.94, CI 0.82–1.08; this was computed ad hoc and must be reproduced in the notebook)
  - [ ] camera / area hotspots during ban windows, normalised per frame
- [ ] **With scraped data**: rain vs truck volume and detection quality; real camera coordinates / official zone
      → hotspots by road
- [ ] Re-run the EDA notebook from `analysis (พัท)/` to confirm the new paths
- [ ] Export 4–6 clean figures for ม่อน (insights) and ฟีฟ่า (slides)
