import argparse
import os
from pathlib import Path

import pandas as pd


def relative(paths, root):
    unique = pd.Series(pd.unique(paths.astype(str)))
    parts = unique.str.replace("\\", "/", regex=False).str.rsplit("/", n=2, expand=True)
    if parts.shape[1] < 3:
        return paths
    new = (root.rstrip("/") + "/" + parts[1] + "/" + parts[2]).where(parts[2].notna(), unique)
    return paths.astype(str).map(dict(zip(unique, new)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--root", default="data/cctv_images")
    ap.add_argument("--col", default="path")
    a = ap.parse_args()
    for f in map(Path, a.files):
        if not f.exists():
            print(f"skip {f}: not found")
            continue
        parquet = f.suffix == ".parquet"
        df = pd.read_parquet(f) if parquet else pd.read_csv(f, dtype=str, keep_default_na=False)
        if a.col not in df.columns:
            print(f"skip {f}: no {a.col} column")
            continue
        new = relative(df[a.col], a.root)
        changed = int((new != df[a.col].astype(str)).sum())
        if changed:
            st = f.stat()
            df[a.col] = new.astype(df[a.col].dtype)
            df.to_parquet(f) if parquet else df.to_csv(f, index=False)
            os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        print(f"{f}: {changed:,} of {len(df):,} paths rewritten to {a.root}/<camera>/<file>")


if __name__ == "__main__":
    main()
