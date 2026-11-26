import argparse
import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import validate  # noqa: E402

STAGES = [
    ("extract", ["data_out/features.csv"]),
    ("cleanse", ["data_out/images_clean.csv", "data_out/cameras.csv"]),
    ("scraped", ["data_out/cameras_geo.csv", "config/ban_zone.geojson", "config/ban_rules.json",
                 "data_out/weather_hourly.csv"]),
    ("detect", ["data_out/detections.parquet"]),
    ("candidates", ["data_out/candidates.parquet"]),
    ("classify", ["data_out/vehicles.parquet"]),
    ("join", ["data_out/context_camera_hour.csv"]),
    ("rules", ["data_out/heavy_sightings.parquet", "data_out/agg_camera_hour.csv"]),
]


@task(task_run_name="make-{target}")
def make(target: str, variables: dict[str, str]) -> None:
    log = get_run_logger()
    cmd = [os.environ.get("MAKE", "make"), "--no-print-directory", target] + [f"{k}={v}" for k, v in variables.items()]
    log.info(" ".join(cmd))
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log.info(line.rstrip())
    if proc.wait():
        raise RuntimeError(f"make {target} exited with {proc.returncode}")


@task(task_run_name="check-{stage}")
def check(stage: str, files: list[str]) -> list[dict]:
    log = get_run_logger()
    os.chdir(ROOT)
    results = validate.run(files)
    for r in results:
        (log.error if r.errors else log.warning if r.warnings else log.info)(r.line())
    failed = [r.path.name for r in results if r.errors]
    if failed:
        raise ValueError(f"data quality check failed after {stage}: {', '.join(failed)}")
    return [dict(stage=stage, **r.row()) for r in results]


@flow(name="bangkok-cctv-truck-ban", log_prints=True)
def pipeline(backend: str = "", force: list[str] | None = None, rescrape: bool = False, py: str = "", py_ml: str = "",
             data: str = "") -> list[dict]:
    variables = {"PY": py or sys.executable}
    variables.update({k: v for k, v in (("BACKEND", backend), ("PY_ML", py_ml), ("DATA", data)) if v})
    rows = []
    for stage, files in STAGES:
        target = "scrape" if stage == "scraped" and rescrape else stage
        extra = {"FORCE": stage} if force and stage in force else {}
        runner = make.with_options(retries=2, retry_delay_seconds=30) if stage == "scraped" else make
        runner(target, variables | extra)
        rows += check(stage, files)
    create_table_artifact(key="data-quality", table=rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="")
    ap.add_argument("--force", nargs="*", default=[], choices=[s for s, _ in STAGES if s != "scraped"])
    ap.add_argument("--rescrape", action="store_true")
    ap.add_argument("--py-ml", default="")
    ap.add_argument("--data", default="")
    a = ap.parse_args()
    pipeline(backend=a.backend, force=a.force, rescrape=a.rescrape, py_ml=a.py_ml, data=a.data)
