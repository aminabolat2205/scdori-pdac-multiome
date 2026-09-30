"""
Merge 11 ATAC samples into one AnnDataSet with a unified peak set, via
whitelist-based direct import (same approach as the 3-sample
snapatac_merge.py, generalized to N samples).

IMPORTANT — two of these samples (10 and 11) share the same original
sample name but come from different pools. That's handled here as follows:

1. SAMPLES is a LIST, not a dict keyed by sample name. A dict keyed by
   sample name would silently collide on sample 10 vs 11 (the second
   entry overwrites the first in a Python dict — sample 10 would just
   vanish with no error). A list can't collide like that.

2. Every SAMPLES entry has an explicit `sample_id` and `pool`, and
   `unique_id = f"{sample_id}__{pool}"` is used everywhere internally
   (AnnDataSet key, obs['sample'], output file names, macs3 groupby).
   validate_samples() checks upfront that no two entries produce the
   same unique_id — so a copy-paste mistake on sample 10/11 raises an
   error immediately instead of silently merging them.

3. obs_names are prefixed with `unique_id` before merging
   ("{unique_id}:{barcode}"). This matters beyond just samples 10/11:
   10x barcode whitelists are shared across GEM wells/pools, so the same
   16bp barcode can legitimately appear in two different pools' data —
   these are two different real cells that coincidentally drew the same
   barcode sequence, not duplicates. Without prefixing, they'd collide
   into literal duplicate obs_names in the merged object. (This was a
   latent bug in the original 3-sample script — it never prefixed
   obs_names, so any "duplicate barcodes" found across samples there were
   most likely genuine distinct cells, not true duplicates. Worth
   revisiting whether cells were wrongly dropped there.) Prefixing here
   makes every obs_name unique by construction, so no dedup-by-barcode
   step should ever be needed downstream.

4. obs['pool'], obs['sample_id'], and obs['sample'] (the unique_id) are
   explicitly re-attached to the final peak matrix after
   make_peak_matrix(), because that function rebuilds a fresh AnnData and
   has been observed to silently drop custom obs columns it doesn't use
   internally (this happened to obs['pool'] in the 3-sample run).

Run in the background:
    nohup python snapatac_merge_11samples.py > merge_11samples.log 2>&1 &
    tail -f merge_11samples.log
"""

import sys
import traceback
from pathlib import Path

import scanpy as sc
import snapatac2 as snap

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = Path(
    "/omics/odcf/analysis/OE0650_projects/saturn3-pdac/amina_jr/snapatac"
)

# EDIT: fill in all 11 samples. sample_id may repeat (as for 10 & 11) —
# pool must be unique per entry, since unique_id = sample_id + pool.
SAMPLES = [
    {
        "sample_id": "S3P-3C1DL-1-T1-S-N0",
        "pool": "pool49",
        "h5ad": "/omics/odcf/.../pool49/atac/S3P-3C1DL-1-T1-S-N0_atac_final.h5ad",
        "fragments": "/omics/odcf/.../atac_frag/pool49_atac_fragments.tsv.gz",
    },
    {
        "sample_id": "S3P-3C1DL-1-M2-S-N0",
        "pool": "pool54",
        "h5ad": "/omics/odcf/.../pool54/atac/S3P-3C1DL-1-M2-S-N0_atac_final.h5ad",
        "fragments": "/omics/odcf/.../atac_frag/pool54_atac_fragments.tsv.gz",
    },
    {
        "sample_id": "S3P-5FP7Z-0-M1-S-N0",
        "pool": "pool33",
        "h5ad": "/omics/odcf/.../pool33/atac/S3P-5FP7Z-0-M1-S-N0_atac_final.h5ad",
        "fragments": "/omics/odcf/.../atac_frag/pool33_atac_fragments.tsv.gz",
    },
    # ... samples 4-9 go here, same shape ...
    {
        "sample_id": "SAME-NAME-EXAMPLE",   # sample 10
        "pool": "poolAA",
        "h5ad": "/omics/odcf/.../poolAA/atac/SAME-NAME-EXAMPLE_atac_final.h5ad",
        "fragments": "/omics/odcf/.../atac_frag/poolAA_atac_fragments.tsv.gz",
    },
    {
        "sample_id": "SAME-NAME-EXAMPLE",   # sample 11 — same sample_id, different pool
        "pool": "poolBB",
        "h5ad": "/omics/odcf/.../poolBB/atac/SAME-NAME-EXAMPLE_atac_final.h5ad",
        "fragments": "/omics/odcf/.../atac_frag/poolBB_atac_fragments.tsv.gz",
    },
]

GENOME = snap.genome.hg38


def make_unique_id(sample_id, pool):
    return f"{sample_id}__{pool}"


def validate_samples(samples):
    seen = {}
    for s in samples:
        uid = make_unique_id(s["sample_id"], s["pool"])
        if uid in seen:
            raise ValueError(
                f"Duplicate (sample_id, pool) combination: {uid!r} — "
                f"two entries in SAMPLES are indistinguishable. Every "
                f"sample needs a unique pool even if sample_id repeats."
            )
        seen[uid] = s
    return seen  # unique_id -> sample info, preserves insertion order


# ============================================================
# FUNCTION: CREATE WHITELIST FROM H5AD
# ============================================================
def create_whitelist(h5ad_file, pool, output_file):
    print(f"Reading H5AD: {h5ad_file}")
    adata = sc.read_h5ad(h5ad_file, backed="r")
    suffix = f"_{pool}"
    barcodes = []
    for barcode in adata.obs_names:
        if not barcode.endswith(suffix):
            raise ValueError(
                f"Unexpected barcode:\n  {barcode}\nExpected suffix: {suffix}"
            )
        # AAACCGAAGTGGCGGA_pool49 -> AAACCGAAGTGGCGGA-1
        barcode = barcode.removesuffix(suffix) + "-1"
        barcodes.append(barcode)
    adata.file.close()
    with open(output_file, "w") as f:
        for barcode in barcodes:
            f.write(barcode + "\n")
    print(f"Number of cells: {len(barcodes)}")
    print(f"Whitelist: {output_file}")
    return len(barcodes)


# ============================================================
# MAIN PIPELINE
# ============================================================
# Wrapped in main() + `if __name__ == "__main__":` because snap.tl.macs3
# (and import_fragments with n_jobs>1) spawn worker subprocesses. Without
# this guard, workers re-execute this whole script from the top instead of
# just importing it — causes "unable to lock file" / worker-died crashes.
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples_by_uid = validate_samples(SAMPLES)
    print(f"{len(samples_by_uid)} samples configured:")
    for uid in samples_by_uid:
        print(f"  {uid}")

    # ============================================================
    # STEP 1: IMPORT EACH SAMPLE
    # ============================================================
    adatas = []
    for uid, info in samples_by_uid.items():
        pool, sample_id = info["pool"], info["sample_id"]

        print("\n" + "=" * 70)
        print(f"PROCESSING {uid}  (pool={pool}, sample_id={sample_id})")
        print("=" * 70)

        whitelist_file = OUTPUT_DIR / f"{uid}_whitelist.txt"
        n_cells = create_whitelist(info["h5ad"], pool, whitelist_file)

        print("\nImporting fragments...")
        print(f"Pool: {pool}")
        print(f"Fragments: {info['fragments']}")
        fragment_output = OUTPUT_DIR / f"{uid}_fragments.h5ad"
        adata = snap.pp.import_fragments(
            info["fragments"],
            chrom_sizes=GENOME,
            whitelist=str(whitelist_file),
            sorted_by_barcode=False,
            is_paired=True,
            min_num_fragments=0,
            file=str(fragment_output),
            n_jobs=8,
        )

        # Prefix obs_names so cross-pool barcode collisions (expected,
        # since 10x barcode whitelists are shared across GEM wells) never
        # produce literal duplicate obs_names once samples are merged.
        adata.obs_names = [f"{uid}:{bc}" for bc in adata.obs_names]

        # Metadata — sample_id can repeat (10 & 11), unique_id never does.
        adata.obs["sample_id"] = [sample_id] * adata.n_obs
        adata.obs["pool"] = [pool] * adata.n_obs
        adata.obs["sample"] = [uid] * adata.n_obs  # unique_id, used for groupby

        print("\nImported object:")
        print(adata)
        print(f"Expected cells: {n_cells} | Imported cells: {adata.n_obs}")
        if adata.n_obs != n_cells:
            print("WARNING: imported cell count differs from H5AD cell count!")

        adatas.append((uid, adata))

    # ============================================================
    # STEP 2: CREATE AnnDataSet
    # ============================================================
    print("\n" + "=" * 70)
    print("CREATING COMBINED AnnDataSet")
    print("=" * 70)
    combined_file = OUTPUT_DIR / "eleven_samples_fragments.h5ads"
    data = snap.AnnDataSet(adatas=adatas, filename=str(combined_file))
    print(data)
    print("\nTotal cells:", data.n_obs)

    if len(set(data.obs_names)) != data.n_obs:
        raise RuntimeError(
            "obs_names are not unique after merging — the uid prefixing "
            "step above should have made this impossible; investigate "
            "before continuing."
        )

    # ============================================================
    # STEP 3: PEAK CALLING (per unique sample/pool combo)
    # ============================================================
    print("\n" + "=" * 70)
    print("CALLING PEAKS")
    print("=" * 70)
    peaks = snap.tl.macs3(data, groupby="sample", inplace=False, n_jobs=8)
    print("Peak calling completed.")

    # ============================================================
    # STEP 4: MERGE PEAKS
    # ============================================================
    print("\n" + "=" * 70)
    print("MERGING PEAKS")
    print("=" * 70)
    merged_peaks = snap.tl.merge_peaks(peaks, GENOME)
    print("Merged peak set:")
    print(merged_peaks)

    # ============================================================
    # STEP 5: CREATE COMMON PEAK MATRIX
    # ============================================================
    print("\n" + "=" * 70)
    print("CREATING COMMON PEAK MATRIX")
    print("=" * 70)
    peak_mat = snap.pp.make_peak_matrix(
        data, use_rep=merged_peaks["Peaks"], inplace=False
    )

    # AnnDataSet only auto-populates 'sample' in its combined .obs (from
    # the tuple keys passed to AnnDataSet(adatas=[(uid, adata), ...])) —
    # 'sample_id'/'pool' were only ever set on the per-sample AnnData
    # objects before merging, so they were never real columns on the
    # combined `data`. Recover them by splitting 'sample' (== unique_id ==
    # f"{sample_id}__{pool}") back apart, rather than looking them up on
    # `data` at all.
    if "sample" not in peak_mat.obs.columns:
        raise RuntimeError(
            "peak_mat has no 'sample' column — expected AnnDataSet to "
            "carry this over automatically."
        )
    sample_ids, pools = [], []
    for uid in peak_mat.obs["sample"]:
        if "__" not in uid:
            raise RuntimeError(f"unique_id {uid!r} missing '__' separator")
        sid, pool = uid.split("__", 1)
        sample_ids.append(sid)
        pools.append(pool)
    peak_mat.obs["sample_id"] = sample_ids
    peak_mat.obs["pool"] = pools

    print("\nFinal peak matrix:")
    print(peak_mat)
    print("Cells:", peak_mat.n_obs)
    print("Peaks:", peak_mat.n_vars)
    print("obs columns:", list(peak_mat.obs.columns))

    # ============================================================
    # STEP 6: SAVE FINAL OBJECT
    # ============================================================
    final_file = OUTPUT_DIR / "eleven_samples_common_peaks.h5ad"
    peak_mat.write_h5ad(str(final_file))

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print(f"Final object: {final_file}")
    print(f"Cells: {peak_mat.n_obs}")
    print(f"Peaks: {peak_mat.n_vars}")

    data.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("FAILED with an exception:", flush=True)
        traceback.print_exc()
        sys.exit(1)
