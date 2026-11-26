"""Rank CCTV road hotspots within the repository's common ban window."""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=HERE, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(ROOT / "data_out/agg_camera_hour.csv")
    hours = frames.loc[frames.ban6].copy()
    assert hours.in_zone.all() and hours.weekday.all() and hours.frames.gt(0).all()
    cameras = pd.read_csv(HERE / "camera_roads.csv")
    assert cameras.camera_id.is_unique
    mapped = hours.merge(cameras, on="camera_id", how="inner", validate="many_to_one")
    rank = mapped.groupby(["road", "road_th"], as_index=False).agg(
        cameras=("camera_id", "nunique"),
        dates=("date", "nunique"),
        frames=("frames", "sum"),
        sightings=("heavy", "sum"),
    )
    rank["sightings"] = rank.sightings.astype(int)
    rank["sightings_per_100_frames"] = 100 * rank.sightings / rank.frames
    rank = rank.sort_values(["sightings_per_100_frames", "road"], ascending=[False, True])
    assert rank.frames.sum() == mapped.frames.sum()
    assert rank.sightings.sum() == mapped.heavy.sum()
    rank.to_csv(args.output / "road_ranking.csv", index=False, float_format="%.4f")
    print(f"Mapped {mapped.camera_id.nunique()}/{hours.camera_id.nunique()} cameras and "
          f"{mapped.frames.sum()}/{hours.frames.sum()} frames; {len(rank)} roads.")
    print(rank.head(8).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = rank.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.road, top.sightings_per_100_frames, color="#00798c")
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=4)
    ax.set_xlim(0, top.sightings_per_100_frames.max() * 1.16)
    ax.set_title("CCTV road hotspots during the project ban window", loc="left")
    ax.set_xlabel("Heavy-truck sightings per 100 frames")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output / "road_hotspots.png", dpi=180)


if __name__ == "__main__":
    main()
