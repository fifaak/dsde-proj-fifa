import argparse
import json
import re
from pathlib import Path

import pandas as pd

RING_ROADS = ["ถนนรัชดาภิเษก", "ถนนจรัญสนิทวงศ์"]
TIME = r"(\d{1,2}[:.]\d{2})\s*[–—-]\s*(\d{1,2}[:.]\d{2})"


def hhmm(t):
    h, m = re.split(r"[:.]", t)
    return f"{int(h):02d}:{m}"


def windows(rows):
    return sorted({(hhmm(a), hhmm(b)) for a, b in zip(rows.start_time, rows.end_time)})


def dump(cfg):
    text = json.dumps(cfg, ensure_ascii=False, indent=2)
    text = re.sub(r"\[\s+((?:\"[^\"\n]*\"|[\d.]+)(?:,\s+(?:\"[^\"\n]*\"|[\d.]+))*)\s+\]",
                  lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", text)
    text = re.sub(r"\[\s+(\[[^\[\]\n]*\](?:,\s+\[[^\[\]\n]*\])*)\s+\]",
                  lambda m: "[" + re.sub(r"\],\s+\[", "], [", m.group(1)) + "]", text)
    return text + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="scraping (เพา)/external_data/data/truck_rules.csv")
    ap.add_argument("--config", default="config/ban_rules.json")
    a = ap.parse_args()
    if not Path(a.rules).exists():
        print(f"{a.rules} not found, {a.config} unchanged")
        return
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    r = pd.read_csv(a.rules, encoding="utf-8-sig", dtype=str).fillna("")
    area = r[r.scope_type.eq("area")]
    heavy = area[area.vehicle_type.str.contains("10 ล้อ")]
    six = area[area.vehicle_type.str.contains("6 ล้อ") & ~area.vehicle_type.str.contains("10 ล้อ")]
    new = json.loads(json.dumps(cfg))
    sources = dict(new.get("sources", {}))
    if len(heavy):
        new["rules"]["truck10"] = [list(w) for w in windows(heavy)]
        sources["truck10"] = " · ".join(sorted(set(heavy.rule_id + ": " + heavy.source_reference)))
        exempt = sorted({(hhmm(x), hhmm(y)) for text in heavy.exceptions for x, y in re.findall(TIME, text)})
        if exempt:
            new["truck10_ring_exempt"] = [list(w) for w in exempt]
            new["ring_roads"] = new.get("ring_roads") or RING_ROADS
            sources["truck10_ring_exempt"] = " · ".join(sorted(set(heavy.rule_id + ": exceptions")))
    if len(six):
        new["rules"]["truck6"] = [list(w) for w in windows(six)]
        sources["truck6"] = " · ".join(sorted(set(six.rule_id + ": " + six.source_reference)))
    else:
        sources.setdefault("truck6", f"no area rule for 6-wheel trucks in {a.rules}; windows kept")
    sources["expressway_rules"] = int(r.scope_type.str.startswith("expressway").sum())
    new["sources"] = sources
    if new != cfg:
        path.write_text(dump(new), encoding="utf-8")
        print(f"{a.config}: updated from {a.rules}")
    print(f"truck6 {new['rules']['truck6']} | truck10 {new['rules']['truck10']} | ring-road exemption "
          f"{new.get('truck10_ring_exempt', [])} on {new.get('ring_roads', [])}")


if __name__ == "__main__":
    main()
