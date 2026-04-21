"""
sync_r2.py — Upload/download dashboard data files to/from Cloudflare R2.

Reads credentials from .env.local (or environment variables):
  R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

Usage:
  python scripts/sync_r2.py upload           # upload all dashboard files to R2
  python scripts/sync_r2.py upload --force   # re-upload even if unchanged
  python scripts/sync_r2.py download         # download all files from R2
  python scripts/sync_r2.py download --missing  # download only files not present locally
  python scripts/sync_r2.py list             # list files in R2 bucket
  python scripts/sync_r2.py status          # compare local vs R2
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

BASE_DIR = Path(__file__).parent.parent

# Files the dashboard needs — relative to BASE_DIR
# Parquet for large pipeline outputs; CSV for small/external-facing files.
DASHBOARD_FILES = [
    # Core ratings — parquet (deploy stays CSV for compat)
    "output/v5_daily_ratings_deploy.csv",
    "output/v5_daily_war.parquet",
    "output/v4_bpm_player_seasons.parquet",
    "output/v5_composite_player_seasons.parquet",
    "output/v5_season_war.parquet",
    "output/v3_rolling_rapm_latest.parquet",
    "output/v6_carry_forward.parquet",
    # Dashboard-ready aggregated (CSV — small/final outputs)
    "output/dashboard_skater_war.csv",
    "output/dashboard_goalie_war.csv",
    "output/dashboard_combined_war.csv",
    "output/win_shares_by_season.parquet",
    # Goalie (small — CSV)
    "output/v2_goalie_war.csv",
    "output/v2_goalie_war_by_season.csv",
    # RAPM — parquet
    "output/v2_rapm_results.csv",
    "output/v2_rapm_by_season.parquet",
    "output/v2_gar_by_season.parquet",
    "output/v2_gar_pooled.parquet",
    # GAR / rating components — parquet
    "output/ifinish_by_season.parquet",
    "output/v3_rolling_rapm.parquet",
    # Team ratings (small — CSV)
    "output/v6_team_season_ratings.csv",
    # JSON weights/calibration
    "output/learned_bpr_weights.json",
    "output/v2_prior_calibration.json",
    # Contracts (CSV — small/external)
    "contracts/active_contracts_by_season.csv",
    "contracts/surplus_values_v2.csv",
    "contracts/career_surplus_v2.csv",
    "contracts/fa_projections_2026.csv",
    "contracts/player_projections_2026.csv",
    "contracts/draft_pick_value_chart.csv",
    "contracts/draft_pick_value_detail.csv",
    "contracts/draft_round_value_chart.csv",
    # Box score data (deploy = trimmed CSV)
    "data/skaters_by_game_deploy.csv",
    "data/skaters_by_game2025_deploy.csv",
    "data/moneypuck_player_bio.csv",
    "data/nhl_draft_picks.csv",
]


def load_env():
    """Load .env.local if present."""
    env_path = BASE_DIR / ".env.local"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())


def get_client():
    load_env()
    endpoint = os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not all([endpoint, access_key, secret_key]):
        print("ERROR: Missing R2 credentials. Set R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY", file=sys.stderr)
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def get_bucket():
    load_env()
    return os.environ.get("R2_BUCKET_NAME", "nhl")


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def list_r2_files(client, bucket):
    """Return dict of {key: etag} for all objects in bucket."""
    objects = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = obj["ETag"].strip('"')
    return objects


def cmd_upload(args):
    client = get_client()
    bucket = get_bucket()

    r2_files = list_r2_files(client, bucket) if not args.force else {}

    uploaded = skipped = missing = 0

    for rel_path in DASHBOARD_FILES:
        local_path = BASE_DIR / rel_path
        if not local_path.exists():
            print(f"  SKIP (not found): {rel_path}")
            missing += 1
            continue

        r2_key = rel_path  # mirror local path structure in R2

        # Skip if already up-to-date (compare MD5 to ETag)
        if not args.force and r2_key in r2_files:
            local_md5 = file_md5(local_path)
            if local_md5 == r2_files[r2_key]:
                print(f"  OK (unchanged): {rel_path}")
                skipped += 1
                continue

        size_mb = local_path.stat().st_size / 1_048_576
        print(f"  UPLOADING ({size_mb:.1f} MB): {rel_path}", end="", flush=True)
        try:
            client.upload_file(str(local_path), bucket, r2_key)
            print(" ✓")
            uploaded += 1
        except ClientError as e:
            print(f" ERROR: {e}")

    print(f"\nDone. Uploaded: {uploaded}, Unchanged: {skipped}, Missing locally: {missing}")


def cmd_download(args):
    client = get_client()
    bucket = get_bucket()

    r2_files = list_r2_files(client, bucket)
    downloaded = skipped = not_found = 0

    for rel_path in DASHBOARD_FILES:
        r2_key = rel_path
        if r2_key not in r2_files:
            print(f"  NOT IN R2: {rel_path}")
            not_found += 1
            continue

        local_path = BASE_DIR / rel_path

        if args.missing and local_path.exists():
            print(f"  SKIP (exists): {rel_path}")
            skipped += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        size_mb = 0
        try:
            head = client.head_object(Bucket=bucket, Key=r2_key)
            size_mb = head["ContentLength"] / 1_048_576
        except ClientError:
            pass

        print(f"  DOWNLOADING ({size_mb:.1f} MB): {rel_path}", end="", flush=True)
        try:
            client.download_file(bucket, r2_key, str(local_path))
            print(" ✓")
            downloaded += 1
        except ClientError as e:
            print(f" ERROR: {e}")

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Not in R2: {not_found}")


def cmd_list(args):
    client = get_client()
    bucket = get_bucket()
    r2_files = list_r2_files(client, bucket)

    if not r2_files:
        print("Bucket is empty.")
        return

    total_bytes = 0
    paginator = client.get_paginator("list_objects_v2")
    objects = {}
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            objects[obj["Key"]] = obj["Size"]
            total_bytes += obj["Size"]

    for key in sorted(objects):
        size_kb = objects[key] / 1024
        print(f"  {size_kb:8.0f} KB  {key}")

    print(f"\n{len(objects)} files, {total_bytes / 1_048_576:.1f} MB total")


def cmd_status(args):
    client = get_client()
    bucket = get_bucket()
    r2_files = list_r2_files(client, bucket)

    print(f"{'File':<55} {'Local':>8} {'R2':>8} {'Status'}")
    print("-" * 85)

    for rel_path in DASHBOARD_FILES:
        local_path = BASE_DIR / rel_path
        local_exists = local_path.exists()
        r2_exists = rel_path in r2_files

        if local_exists:
            size_kb = f"{local_path.stat().st_size / 1024:.0f}KB"
        else:
            size_kb = "—"

        if r2_exists:
            r2_size = "✓"
        else:
            r2_size = "—"

        if local_exists and r2_exists:
            status = "synced"
        elif local_exists and not r2_exists:
            status = "UPLOAD NEEDED"
        elif not local_exists and r2_exists:
            status = "download available"
        else:
            status = "missing everywhere"

        print(f"  {rel_path:<53} {size_kb:>8} {r2_size:>8}  {status}")


def main():
    parser = argparse.ArgumentParser(description="Sync dashboard data files with Cloudflare R2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Upload dashboard files to R2")
    up.add_argument("--force", action="store_true", help="Re-upload even if unchanged")

    dl = sub.add_parser("download", help="Download dashboard files from R2")
    dl.add_argument("--missing", action="store_true", help="Download only files not present locally")

    sub.add_parser("list", help="List files in R2 bucket")
    sub.add_parser("status", help="Compare local files with R2")

    args = parser.parse_args()

    try:
        if args.cmd == "upload":
            cmd_upload(args)
        elif args.cmd == "download":
            cmd_download(args)
        elif args.cmd == "list":
            cmd_list(args)
        elif args.cmd == "status":
            cmd_status(args)
    except NoCredentialsError:
        print("ERROR: Could not find AWS credentials.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
