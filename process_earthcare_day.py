#!/usr/bin/env python3
"""
Download EarthCARE ACM_CAP_2B data, bin ice_water_path onto a HEALPix
pyramid (levels 0-11), and save the pyramid to disk — for a single day
or a range of days.

Usage
-----
Single day:
    python process_earthcare_day.py --date 2024-08-10

Range of days (inclusive), processed sequentially in one run:
    python process_earthcare_day.py --start-date 2024-08-10 --end-date 2024-09-30
"""

import argparse
import datetime as dt
import logging
import shutil
import sys
from pathlib import Path

import earthcarekit as eck
import grid_doctor as gd

PRODUCT = "ACM_CAP_2B"
OUTPUT_ROOT = Path("/work/ks1387/gw/data/obs/earthcare")
RAW_DATA_DIR = Path("/home/k/k204228/EarthCARE/data")
MAX_LEVEL = 11
MIN_LEVEL = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def daterange(start: dt.date, end: dt.date):
    """Yield each date from start to end, inclusive."""
    for n in range((end - start).days + 1):
        yield start + dt.timedelta(days=n)


def clean_raw_data_dir() -> None:
    """Remove everything ecdownload has written to the raw data directory.

    This runs before each new day starts downloading, so raw files never
    accumulate across days and fill up the home directory.
    """
    if not RAW_DATA_DIR.exists():
        return

    removed = 0
    for entry in RAW_DATA_DIR.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except Exception:
            log.exception("Could not remove %s", entry)

    if removed:
        log.info("Cleared %d item(s) from raw data dir %s", removed, RAW_DATA_DIR)


def process_day(day: dt.date) -> None:
    day_str = day.strftime("%Y-%m-%d")
    start_time = f"{day_str}T00:00:00"
    end_time = (day + dt.timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    out_dir = OUTPUT_ROOT / f"{PRODUCT}_{day_str}"

    if out_dir.exists() and any(out_dir.iterdir()):
        log.info("Output for %s already exists, skipping.", day_str)
        return

    clean_raw_data_dir()

    log.info("Downloading %s for %s", PRODUCT, day_str)
    eck.ecdownload(PRODUCT, start_time=start_time, end_time=end_time)

    products = eck.search_product(file_type=PRODUCT)
    if len(products) == 0:
        log.warning("No %s product found for %s, skipping.", PRODUCT, day_str)
        return

    ds = eck.read_product(products.filepath[0])

    log.info("Binning to HEALPix level %d for %s", MAX_LEVEL, day_str)
    finest = gd.bin_to_healpix(
        ds[["ice_water_path", "longitude", "latitude"]],
        level=MAX_LEVEL,
        agg="mean",
        lat_name="latitude",
        lon_name="longitude",
    )

    pyramid = {MAX_LEVEL: finest}
    for level in range(MAX_LEVEL - 1, MIN_LEVEL - 1, -1):
        pyramid[level] = gd.coarsen_healpix(pyramid[level + 1], level)

    out_dir.mkdir(parents=True, exist_ok=True)
    gd.save_pyramid(pyramid, str(out_dir) + "/")
    log.info("Saved pyramid for %s to %s", day_str, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", type=str, help="Single date, YYYY-MM-DD")
    group.add_argument(
        "--start-date", type=str, help="Start date, YYYY-MM-DD (used with --end-date)"
    )
    parser.add_argument(
        "--end-date", type=str, help="End date, YYYY-MM-DD, inclusive (used with --start-date)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.date:
        days = [dt.date.fromisoformat(args.date)]
    else:
        if not args.end_date:
            sys.exit("--end-date is required when using --start-date")
        days = list(
            daterange(
                dt.date.fromisoformat(args.start_date),
                dt.date.fromisoformat(args.end_date),
            )
        )

    log.info("Processing %d day(s): %s ... %s", len(days), days[0], days[-1])

    failures = []
    for day in days:
        try:
            process_day(day)
        except Exception:
            log.exception("Failed to process %s", day.strftime("%Y-%m-%d"))
            failures.append(day.strftime("%Y-%m-%d"))
            continue

    if failures:
        log.warning("Finished with %d failed day(s): %s", len(failures), ", ".join(failures))
        sys.exit(1)

    log.info("All days processed successfully.")


if __name__ == "__main__":
    main()
