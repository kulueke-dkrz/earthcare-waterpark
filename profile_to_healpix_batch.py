import earthcarekit as eck
import numpy as np
import xarray as xr
import xdggs
import zarr

from earthcare_dggs.convert import lonlat_to_healpix_cells
from earthcare_dggs.settings import ATLID_DEPTH, ELLIPSOID

PRODUCT = "ACM_CAP_2B"
ORBITS = ["01164E", "01164F", "01165A"]  # orbit_and_frame ids to process
OUTPUT_PATH = ""                          # prefix for output store paths
SKIP_EXISTING = True                      # skip an orbit if its store already exists
VERBOSE = False                           # print full per-variable shape listing


def output_path_for(orbit):
    return f"{OUTPUT_PATH}earthcare_{PRODUCT}_{orbit}.zarr"


def store_exists(path):
    """Check whether a Zarr store already exists at path (no os/pathlib needed)."""
    try:
        zarr.open_group(path, mode="r")
        return True
    except Exception:
        return False


def write_dggs_convention(output_path):
    """Add DGGS Zarr convention metadata (compatible with legacy-converters format)."""
    dggs_convention = {
        "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38",
        "name": "dggs",
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
        "description": "Discrete Global Grid Systems convention for zarr",
    }
    dggs_meta = {
        "name": "healpix",
        "refinement_level": ATLID_DEPTH,
        "indexing_scheme": "nested",
        "ellipsoid": {
            "name": "wgs84",
            "semimajor_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
        "spatial_dimension": "cell_ids",
        "coordinate": "cell_ids",
        "compression": "none",
    }
    root = zarr.open_group(output_path, mode="r+")
    root.attrs["zarr_conventions"] = [dggs_convention]
    root.attrs["dggs"] = dggs_meta
    print(f"DGGS metadata: {dggs_meta}")


def process_segment(orbit):
    """Load, HEALPix-index, and write a single EarthCARE segment to Zarr.

    Returns a result dict describing what happened, so the batch loop
    can summarize successes/skips/failures at the end.
    """
    output_path = output_path_for(orbit)

    if SKIP_EXISTING and store_exists(output_path):
        print(f"[{orbit}] output already exists, skipping ({output_path})")
        return {"orbit": orbit, "status": "skipped", "output_path": output_path}

    result = eck.search_product(file_type=PRODUCT, orbit_and_frame=orbit)
    if len(result.filepath) == 0:
        print(f"[{orbit}] no matching product file found, skipping")
        return {"orbit": orbit, "status": "not_found"}
    if len(result.filepath) > 1:
        print(f"[{orbit}] {len(result.filepath)} files matched, using the first: {result.filepath[0]}")

    with eck.read_product(result.filepath[0]) as ds:
        profile = ds.load()

    n_profiles = len(profile.along_track)
    print(f"[{orbit}] Dimensions: {dict(profile.sizes)}")
    print(f"[{orbit}] Lat: [{float(profile['latitude'].min()):.1f}, {float(profile['latitude'].max()):.1f}]")
    print(f"[{orbit}] Lon: [{float(profile['longitude'].min()):.1f}, {float(profile['longitude'].max()):.1f}]")
    if VERBOSE:
        print(f"[{orbit}] 2D variables (along_track, vertical):")
        for v in profile.data_vars:
            if profile[v].dims == ("along_track", "vertical"):
                print(f"  {v}: shape={profile[v].shape}")

    prof_cell_ids = lonlat_to_healpix_cells(
        profile["longitude"].values, profile["latitude"].values,
        depth=ATLID_DEPTH, ellipsoid=ELLIPSOID,
    )

    unique_cells = np.unique(prof_cell_ids)
    n_cells = len(unique_cells)
    print(f"[{orbit}] HEALPix depth {ATLID_DEPTH}: {n_profiles} profiles → {n_cells} unique cells")
    print(f"[{orbit}] Average profiles per cell: {n_profiles / n_cells:.1f}")

    # Add cell_ids as coordinate
    profile_healpix = profile.assign_coords(cell_ids=("along_track", prof_cell_ids))
    profile_healpix.attrs.update({
        "source": "EarthCARE MSI L2A",
        "healpix_level": ATLID_DEPTH,
        "healpix_indexing": "nested",
        "healpix_ellipsoid": ELLIPSOID,
        "orbit_and_frame": orbit,
    })

    # Add xdggs index
    profile_healpix = xdggs.decode(
        profile_healpix,
        grid_info=xdggs.HealpixInfo(level=ATLID_DEPTH, indexing_scheme="nested"),
    )

    # Encode with xdggs convention
    profile_encoded = xdggs.encode(profile_healpix, "xdggs")

    # Write to Zarr
    profile_encoded.to_zarr(output_path, mode="w", consolidated=True)

    # Add DGGS Zarr convention metadata
    write_dggs_convention(output_path)

    # Nothing needs to stay resident once the store is written, since
    # segments aren't combined -- drop references before the next segment.
    del profile, profile_healpix, profile_encoded

    print(f"[{orbit}] Saved to {output_path}")
    return {
        "orbit": orbit,
        "status": "ok",
        "output_path": output_path,
        "n_profiles": n_profiles,
        "n_cells": n_cells,
    }


def main():
    results = []
    for i, orbit in enumerate(ORBITS, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(ORBITS)}] orbit_and_frame={orbit}")
        print(f"{'=' * 60}")
        try:
            results.append(process_segment(orbit))
        except Exception as e:
            print(f"[{orbit}] ERROR: {e}")
            results.append({"orbit": orbit, "status": "failed", "error": str(e)})

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    not_found = [r for r in results if r["status"] == "not_found"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n{'=' * 60}")
    print(f"Batch summary: {len(results)} orbits requested")
    print(f"  ok:        {len(ok)}")
    print(f"  skipped:   {len(skipped)}  (already existed)")
    print(f"  not found: {len(not_found)}")
    print(f"  failed:    {len(failed)}")
    print(f"{'=' * 60}")

    if ok:
        total_profiles = sum(r["n_profiles"] for r in ok)
        total_cells = sum(r["n_cells"] for r in ok)
        print(f"Newly converted: {total_profiles} profiles, {total_cells} cell entries (not globally deduped)")
        for r in ok:
            print(f"  {r['orbit']}: {r['n_profiles']} profiles, {r['n_cells']} cells -> {r['output_path']}")

    if not_found:
        print(f"Not found: {', '.join(r['orbit'] for r in not_found)}")
    if failed:
        print("Failed:")
        for r in failed:
            print(f"  {r['orbit']}: {r['error']}")

    return results


if __name__ == "__main__":
    main()
