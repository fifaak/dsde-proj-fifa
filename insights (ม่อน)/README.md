# insights (ม่อน)/ — findings

Write-up of what the analysis shows, for the slides and the video. Raw numbers are in the root `README.md`
("First results"); figures can be exported from the dashboard or `analysis (พัท)/`.

## Evidence images

`evidence/` = 49 CCTV frames of heavy trucks **inside the ban zone during weekday ban windows** (top 10 cameras,
up to 5 different ban windows each, model confidence p ≥ 0.95), trucks boxed in red, with a caption.
`evidence/index.csv` lists camera, place, time and confidence; `evidence/contact_sheet.jpg` shows all of them.
Regenerate with `python "insights (ม่อน)/make_evidence.py"` (needs the raw images locally).
These are model detections, not verified violations: check each image before using it as an example.
The full list of 3,014 sightings is `data_out/heavy_sightings.parquet` (rows with `in_ban6 == True`).

## Draft findings (to refine)

1. **Trucks follow the daily rhythm, not the ban.** In-zone weekday ban hours vs 10–14h, compared with the weekend:
   ratio 0.94 (95% CI 0.82–1.08) — no clear extra drop during ban windows.
2. **Hotspots.** Most in-zone ban-window sightings: cameras 11, 75, 68, 26, 32, 188 (check camera 11: passenger vans).
3. **Volume.** 3,014 heavy-truck sightings inside the zone during weekday ban windows (≈ 1 frame / 10 min / camera).

## Limitations to state

- 7 days only (25–31 Aug 2026), 2 weekend days, 06:00–21:00, one snapshot every ~10 min.
- Heavy-truck classifier precision ≈ 0.65 / recall ≈ 0.69 (90-label holdout); 6 vs 10 wheels not separable.
- Ban zone and camera positions are approximate until the official sources are scraped.
- System outage 27 Aug 06:00–08:50; weaker detection after ~18:30.

## TODO

- Join weather (rain) and official rules once `scraping (เพา)/` delivers them.
- 3–5 headline insights + recommendations (where/when to enforce).
