#!/usr/bin/env python
"""Pair up RNA and ATAC AnnData objects so they can go into scDoRI.

Inputs:
    ATAC  peak matrix from the SnapATAC2 merge step. Barcodes look like
          'AAACCGAAGTGGCGGA-1' and obs['sample'] says which sample each
          cell came from.
    RNA   merged RNA object for the same samples. Barcodes look like
          'AAACCGAAGTGGCGGA_<pool>'.

scDoRI needs both modalities as .h5ad files with raw counts in .X, the exact
same cells in the same order, and chr/start/end columns in the ATAC .var
(see https://scdori.readthedocs.io/en/latest/).

What this script does:
    1. Renames ATAC barcodes from '{barcode}-1' to '{barcode}_{pool}' so they
       match the RNA naming.
    2. Keeps only the cells present in both objects, in the same order.
    3. Adds chr/start/end columns to the ATAC .var from the peak names.
    4. Warns if either .X does not look like raw counts.
    5. Writes the two paired files.

Set the paths and the sample to pool mapping below, then run:
    python prepare_inputs.py
"""

import re
import sys

import anndata as ad
import numpy as np

# Input and output files
ATAC_IN = "atac_common_peaks.h5ad"
RNA_IN = "merged_rna.h5ad"
RNA_OUT = "rna_scdori_input.h5ad"
ATAC_OUT = "atac_scdori_input.h5ad"

# Which pool each ATAC sample belongs to. This has to match the pool suffix
# used in the RNA barcodes. The pool column itself gets dropped when
# SnapATAC2 builds the peak matrix, so it is rebuilt from obs['sample'] here.
# If obs['pool'] is still present, it is used directly and this is ignored.
SAMPLE_TO_POOL = {
    "sample1": "pool1",
    "sample2": "pool2",
    "sample3": "pool3",
}

PEAK_PATTERN = re.compile(r"^([\w.]+)[:_-](\d+)[-_](\d+)$")


def log(message):
    print(message, flush=True)


def get_pools(atac):
    """Return the pool label for every ATAC cell."""
    if "pool" in atac.obs.columns:
        return atac.obs["pool"].astype(str).tolist()

    if "sample" not in atac.obs.columns:
        raise RuntimeError(
            "The ATAC object has neither obs['pool'] nor obs['sample'], so "
            "there is no way to rebuild barcodes that match the RNA object."
        )

    unknown = set(atac.obs["sample"].unique()) - set(SAMPLE_TO_POOL)
    if unknown:
        raise RuntimeError(
            f"These samples are missing from SAMPLE_TO_POOL: {sorted(unknown)}. "
            "Add them to the mapping at the top of the script."
        )
    return [SAMPLE_TO_POOL[s] for s in atac.obs["sample"]]


def retag_atac_barcodes(atac):
    """Turn '{barcode}-1' into '{barcode}_{pool}'."""
    barcodes = [name.rsplit("-", 1)[0] for name in atac.obs_names]
    pools = get_pools(atac)
    atac.obs_names = [f"{bc}_{pool}" for bc, pool in zip(barcodes, pools)]
    return atac


def parse_peak_coords(peak_names):
    """Split peak names like 'chr1:100-200' into chromosome, start and end.

    'chr1-100-200' and 'chr1_100_200' work too.
    """
    chroms, starts, ends = [], [], []
    for name in peak_names:
        match = PEAK_PATTERN.match(name)
        if match is None:
            raise ValueError(f"Can't read peak coordinates from {name!r}")
        chrom, start, end = match.groups()
        chroms.append(chrom)
        starts.append(int(start))
        ends.append(int(end))
    return chroms, starts, ends


def looks_like_raw_counts(adata, n_rows=100):
    """Quick check on the first rows: non-negative whole numbers."""
    block = adata.X[:n_rows]
    block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
    return block.min() >= 0 and np.allclose(block, np.round(block))


def main():
    log(f"Loading ATAC from {ATAC_IN}")
    atac = ad.read_h5ad(ATAC_IN)
    log(f"  {atac.n_obs} cells, {atac.n_vars} peaks")

    log(f"Loading RNA from {RNA_IN}")
    rna = ad.read_h5ad(RNA_IN)
    log(f"  {rna.n_obs} cells, {rna.n_vars} genes")

    atac = retag_atac_barcodes(atac)

    if atac.obs_names.duplicated().any():
        raise RuntimeError(
            "Some ATAC barcodes are duplicated after renaming. Check that "
            "every sample maps to the right pool."
        )
    if rna.obs_names.duplicated().any():
        raise RuntimeError("Some RNA barcodes are duplicated.")

    shared = sorted(set(atac.obs_names) & set(rna.obs_names))
    log(f"Cells in ATAC: {atac.n_obs}, in RNA: {rna.n_obs}, in both: {len(shared)}")

    if not shared:
        print(
            "No cells in common. Compare the barcode formats by hand:\n"
            f"  ATAC: {atac.obs_names[0]!r}\n"
            f"  RNA:  {rna.obs_names[0]!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    atac = atac[shared].copy()
    rna = rna[shared].copy()

    chroms, starts, ends = parse_peak_coords(atac.var_names)
    atac.var["chr"] = chroms
    atac.var["start"] = starts
    atac.var["end"] = ends

    for label, adata in (("RNA", rna), ("ATAC", atac)):
        if not looks_like_raw_counts(adata):
            print(
                f"Warning: {label}.X doesn't look like raw counts (negative or "
                "non-integer values). scDoRI needs raw counts, so make sure a "
                "normalized or log layer didn't end up in .X.",
                file=sys.stderr,
            )

    rna.write_h5ad(RNA_OUT)
    atac.write_h5ad(ATAC_OUT)

    log(f"RNA:  {rna.n_obs} cells, {rna.n_vars} genes  -> {RNA_OUT}")
    log(f"ATAC: {atac.n_obs} cells, {atac.n_vars} peaks -> {ATAC_OUT}")


if __name__ == "__main__":
    main()
