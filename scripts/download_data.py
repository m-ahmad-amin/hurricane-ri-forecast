from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download import download_basin_season, download_storm
from src.utils import ensure_dirs, load_config, project_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TC PRIMED from the public NOAA S3 bucket")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--basin", type=str, required=True, help="AL or EP")
    parser.add_argument("--storm", type=str, default=None, help="optional storm number like 06")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    ensure_dirs(paths)

    bucket = cfg["download"]["bucket"]
    version = cfg["download"]["version"]
    stage = cfg["download"]["stage"]
    raw_root = paths["raw_tcprimed"]

    if args.storm:
        out = download_storm(
            bucket, version, stage, args.season, args.basin.upper(), args.storm, raw_root
        )
    else:
        out = download_basin_season(
            bucket, version, stage, args.season, args.basin.upper(), raw_root
        )
    print(f"downloaded into {out}")


if __name__ == "__main__":
    main()
