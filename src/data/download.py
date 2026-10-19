from __future__ import annotations

from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


def make_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_keys(bucket: str, prefix: str) -> list[dict]:
    s3 = make_s3_client()
    keys: list[dict] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys.extend(resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return keys


def download_prefix(
    bucket: str,
    prefix: str,
    out_dir: Path,
    skip_existing: bool = True,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3 = make_s3_client()
    objects = list_keys(bucket, prefix)
    downloaded = 0
    for obj in tqdm(objects, desc=f"download {prefix}"):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        rel = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and dest.exists() and dest.stat().st_size == obj["Size"]:
            continue
        s3.download_file(bucket, key, str(dest))
        downloaded += 1
    return downloaded


def download_storm(
    bucket: str,
    version: str,
    stage: str,
    season: int,
    basin: str,
    storm_id: str,
    raw_root: Path,
) -> Path:
    prefix = f"{version}/{stage}/{season}/{basin}/{storm_id}/"
    out_dir = Path(raw_root) / str(season) / basin / storm_id
    download_prefix(bucket, prefix, out_dir)
    return out_dir


def download_basin_season(
    bucket: str,
    version: str,
    stage: str,
    season: int,
    basin: str,
    raw_root: Path,
) -> Path:
    prefix = f"{version}/{stage}/{season}/{basin}/"
    out_dir = Path(raw_root) / str(season) / basin
    download_prefix(bucket, prefix, out_dir)
    return out_dir
