"""Approximate camera coordinates by geocoding the landmark in each camera name with OpenStreetMap Nominatim.

The BMA Traffic site (source of the camera list) was not reachable from this machine, so each camera gets a
hand-written landmark query (intersection / footbridge landmark / soi) parsed from its Thai name. Accuracy is
landmark-level (~50-300 m), good enough for a city map, not for lane-level placement. Replace with the official
BMA camera coordinates when available (same output format).

Usage: python scraping/geocode_cameras.py  -> data_out/cameras_geo.csv (camera_id, lat, lon, query, osm_name, method)
Nominatim policy: max 1 request/s, results cached in data_out/geocode_cache.json.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# camera_id -> landmark queries tried in order (first hit inside Bangkok wins)
QUERIES = {
    11: ["ซอยราชวิถี 6", "อนุสาวรีย์ชัยสมรภูมิ"], 18: ["แยกสะพานแดง"], 26: ["แยกบุคคโล"], 32: ["แยกบางยี่เรือ"],
    36: ["แยกรัชดา-ราชพฤกษ์", "แยกรัชดาราชพฤกษ์"], 68: ["แยกแม้นศรี"], 70: ["ธนาคารแห่งประเทศไทย"],
    75: ["แยกรัชดา-นราธิวาส", "แยกนราธิวาส-รัชดา"], 82: ["ซอยราชปรารภ 3", "แยกหมอเหล็ง"],
    88: ["แยกจันทน์-สาธุประดิษฐ์", "แยกจันทน์"], 112: ["แยกเทียมร่วมมิตร"], 118: ["แยกรัชดา-สุทธิสาร", "แยกสุทธิสาร"],
    119: ["แยกรัชดา-สุทธิสาร", "แยกสุทธิสาร"], 120: ["แยกรัชดา-สุทธิสาร", "แยกสุทธิสาร"],
    123: ["แยกรัชดา-ลาดพร้าว", "สถานีลาดพร้าว"], 124: ["แยกรัชดา-ลาดพร้าว", "สถานีลาดพร้าว"],
    125: ["แยกรัชดา-ลาดพร้าว", "สถานีลาดพร้าว"], 126: ["ซอยรัชดาภิเษก 30"], 127: ["ซอยรัชดาภิเษก 30"],
    135: ["โรงเรียนหอวัง", "แยกรัชโยธิน"], 136: ["Big C Extra Ladprao", "บิ๊กซี ลาดพร้าว"],
    137: ["Big C Extra Ladprao", "บิ๊กซี ลาดพร้าว"], 139: ["โรงเรียนหอวัง"], 141: ["PTT Head Office", "ห้าแยกลาดพร้าว"],
    150: ["โรงเรียนสามเสนวิทยาลัย"], 158: ["ซอยเจริญนคร 46"], 169: ["วัดเสาหิน", "สะพานพระราม 7"],
    170: ["โรงพยาบาลราชวิถี", "อนุสาวรีย์ชัยสมรภูมิ"], 176: ["สวนจตุจักร"], 177: ["สวนจตุจักร"],
    178: ["โรงพยาบาลพญาไท 1"], 179: ["โรงพยาบาลพญาไท 1"], 186: ["แยกรัชดา-นราธิวาส"],
    188: ["แยก ณ ระนอง", "ถนนพระรามที่ 3"], 194: ["ซอยคอนแวนต์"], 196: ["Makro Sathorn", "แยกสาทร-นราธิวาส"],
    206: ["แยกรัชดา-นราธิวาส"], 211: ["ซอยรัชดาภิเษก 3"], 212: ["Big C Ratchadaphisek", "บิ๊กซี รัชดาภิเษก"],
    215: ["ซอยรัชดาภิเษก 7"], 219: ["ซอยรัชดาภิเษก 14"], 224: ["ศาลอาญา"], 225: ["ศาลอาญา"],
    228: ["โรงพยาบาลกลาง"], 230: ["ซอยระนอง 2"], 233: ["พระตำหนักจิตรลดารโหฐาน"], 239: ["โรงเรียนเตรียมอุดมศึกษา"],
    250: ["ซอยเพชรบุรี 38"], 251: ["วัดดิสหงษาราม"], 256: ["ซอยเพชรบุรี 27"], 277: ["อนุสาวรีย์ชัยสมรภูมิ"],
    301: ["แยกอู่ทองนอก"], 332: ["แยกราชวัตร"], 363: ["สำนักงานเขตดุสิต"], 370: ["แยกเทเวศร์", "แยกเทเวศน์"],
    371: ["แยกเทเวศร์", "แยกเทเวศน์"], 395: ["เสาชิงช้า"], 411: ["ถนนพระอาทิตย์"], 412: ["ถนนพระอาทิตย์"],
    413: ["ถนนตะนาว"], 414: ["สะพานพระพุทธยอดฟ้า"], 432: ["แยกสามย่าน"], 500: ["แยกมิตรสัมพันธ์", "แยกราชประสงค์"],
    529: ["ถนนพุทธมณฑลสาย 2"], 542: ["แยกหลังสวน"], 546: ["แยกพาณิชยการพระนคร", "พาณิชยการพระนคร"],
    556: ["แยกวชิระ", "โรงพยาบาลวชิรพยาบาล"], 569: ["แยก จปร", "แยก จ.ป.ร."], 572: ["ตลิ่งชัน บรมราชชนนี", "สถานีตลิ่งชัน"],
    576: ["แยกบางลำพู"], 595: ["แยกบ้านแขก"], 599: ["แยกสุรศักดิ์"], 603: ["แยกสีลม-นราธิวาส", "แยกนราธิวาส-สีลม"],
    604: ["แยกสีลม-นราธิวาส", "แยกนราธิวาส-สีลม"], 619: ["แยกอังรีดูนังต์"], 630: ["แยกเกษะโกมล"],
    913: ["แยก S.A.B.", "แยกเอสเอบี"], 915: ["วงเวียนใหญ่"], 938: ["แยกพระราม 9-รามคำแหง", "แยกรามคำแหง"],
    939: ["แยกมหาดไทย"], 1028: ["แยกสะพานแดง"], 1039: ["สะพานนริศดำรัส"], 1060: ["โรงเรียนสายปัญญา"],
    1063: ["พระตำหนักจิตรลดารโหฐาน"], 1070: ["ซอยพหลโยธิน 18/2", "แยกกำแพงเพชร"], 1071: ["แยกกำแพงเพชร"],
    1076: ["ซอยเทอดไท 21", "ถนนเทอดไท"], 1078: ["วัดสุวรรณ", "แยกคลองสาน"], 1103: ["แยกรัชดา-ราชพฤกษ์", "แยกท่าพระ"],
    1112: ["แยกสุทธิสาร"], 1132: ["วัดสุวรรณ", "แยกคลองสาน"], 1188: ["ถนนพุทธมณฑลสาย 2"], 1195: ["แยกบุคคโล"],
    1361: ["ธนาคารแห่งประเทศไทย"], 1362: ["แยกสามเสน"], 1368: ["แยกพาณิชยการพระนคร", "พาณิชยการพระนคร"],
    1408: ["แยกเทียมร่วมมิตร"], 1589: ["แยกแม้นศรี"], 1804: ["ซอยอ่อนนุช 33"], 1815: ["แยกพระราม 9"],
}
# corrections after manual review of the first run (wrong-place hits)
QUERIES.update({
    32: ["Bang Yi Ruea Intersection", "ตลาดพลู"], 88: ["Chan Sathu Pradit Intersection", "ถนนสาธุประดิษฐ์"],
    112: ["Thiam Ruam Mit Intersection", "สถานีศูนย์วัฒนธรรมแห่งประเทศไทย"], 1408: ["Thiam Ruam Mit Intersection", "สถานีศูนย์วัฒนธรรมแห่งประเทศไทย"],
    136: ["Big C Extra Lat Phrao", "สถานีพหลโยธิน"], 137: ["Big C Extra Lat Phrao", "สถานีพหลโยธิน"],
    212: ["Big C Supercenter Ratchadaphisek", "สถานีห้วยขวาง"], 529: ["Phutthamonthon Sai 2 Borommaratchachonnani", "ตลิ่งชัน"],
    1188: ["Phutthamonthon Sai 2 Borommaratchachonnani", "ตลิ่งชัน"], 556: ["วชิรพยาบาล", "Vajira Hospital"],
    939: ["กระทรวงมหาดไทย"], 1076: ["ตลาดพลู", "สถานีตลาดพลู"], 1815: ["สถานีพระราม 9", "Phra Ram 9 MRT"],
    75: ["Ratchada-Narathiwat Intersection", "ถนนนราธิวาสราชนครินทร์ พระราม 3", "ถนนนราธิวาสราชนครินทร์"],
    186: ["Ratchada-Narathiwat Intersection", "ถนนนราธิวาสราชนครินทร์"], 206: ["Ratchada-Narathiwat Intersection", "ถนนนราธิวาสราชนครินทร์"],
    196: ["สถานีช่องนนทรี", "Chong Nonsi"], 603: ["สถานีช่องนนทรี", "Chong Nonsi"], 604: ["สถานีช่องนนทรี", "Chong Nonsi"],
    546: ["วัดเบญจมบพิตรดุสิตวนาราม"], 1368: ["วัดเบญจมบพิตรดุสิตวนาราม"],
    529: ["ถนนพุทธมณฑลสาย 2"], 1188: ["ถนนพุทธมณฑลสาย 2"],
    112: ["ศูนย์วัฒนธรรมแห่งประเทศไทย", "Thailand Cultural Centre"], 1408: ["ศูนย์วัฒนธรรมแห่งประเทศไทย", "Thailand Cultural Centre"],
    1815: ["เซ็นทรัล พระราม 9", "Central Rama 9", "ฟอร์จูนทาวน์"],
})
# landmark-level approximations (road point / nearby landmark rather than the exact junction)
APPROX = {32, 75, 88, 136, 137, 186, 196, 206, 212, 529, 546, 556, 603, 604, 1076, 1188, 1368}
BKK = (13.49, 100.32, 13.96, 100.94)          # lat_min, lon_min, lat_max, lon_max
UA = "cctv-course-project/0.1 (educational, low volume)"
CACHE = Path("data_out/geocode_cache.json")


def search(q, cache):
    if q in cache:
        return cache[q]
    params = {"q": q, "format": "jsonv2", "limit": 1, "countrycodes": "th", "bounded": 1,
              "viewbox": f"{BKK[1]},{BKK[2]},{BKK[3]},{BKK[0]}"}
    req = urllib.request.Request("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    time.sleep(1.1)
    cache[q] = res[0] if res else None
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    return cache[q]


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cams = pd.read_csv("data_out/cameras.csv")
    rows = []
    for cid in cams.camera_id:
        hit, used = None, None
        for i, q in enumerate(QUERIES.get(cid, [])):
            hit = search(q, cache)
            if hit:
                used = (q, "primary" if i == 0 else "fallback")
                break
        rows.append({"camera_id": cid, "lat": float(hit["lat"]) if hit else None,
                     "lon": float(hit["lon"]) if hit else None, "query": used[0] if used else None,
                     "osm_name": hit.get("display_name", "")[:80] if hit else None,
                     "method": (used[1] if used else "not_found") + ("_approx" if cid in APPROX and used else "")})
    geo = pd.DataFrame(rows)
    geo.to_csv("data_out/cameras_geo.csv", index=False)
    print(geo.method.value_counts().to_dict())
    print(geo[geo.method != "primary"][["camera_id", "query", "method"]].to_string(index=False))


if __name__ == "__main__":
    main()
