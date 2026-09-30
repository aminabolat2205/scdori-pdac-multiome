#!/usr/bin/env python
"""Preprocess paired scRNA-seq and scATAC-seq data for scDoRI.

Steps:
    1. Set up output directories and download genome references
    2. Load RNA and ATAC AnnData objects and keep shared, QC-passing cells
    3. Select highly variable genes and TFs
    4. Restrict peaks to a window around the selected genes
    5. Build metacells and keep promoter peaks plus highly variable peaks
    6. Score TF motifs in the selected peaks (FIMO via tangermeme)
    7. Compute in silico ChIP-seq scores
    8. Compute gene-peak distances and the distance decay prior

All settings are read from scdori.pp.ppConfig.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scdori.pp import (
    ppConfig,
    compute_gene_peak_distance_matrix,
    compute_hvgs_and_tfs,
    compute_in_silico_chipseq,
    compute_motif_scores,
    create_dir_if_not_exists,
    create_extended_gene_bed,
    create_metacells,
    download_genome_references,
    filter_protein_coding_genes,
    intersect_cells,
    keep_promoters_and_select_hv_peaks,
    load_anndata,
    load_gtf,
    load_motif_database,
    remove_mitochondrial_genes,
    run_bedtools_intersect,
    save_processed_datasets,
)

logger = logging.getLogger(__name__)

# Promoter window around the TSS (strand aware). This is the usual ATAC-seq
# promoter definition and is separate from the wider gene body window used
# later for the gene-peak distance prior.
PROMOTER_UPSTREAM = 2000
PROMOTER_DOWNSTREAM = 500


def add_peak_coordinates(adata):
    """Parse 'chr:start-end' peak names into chr/start/end/peak_name columns."""
    names = adata.var_names
    adata.var["chr"] = [name.split(":")[0] for name in names]
    adata.var["start"] = [int(name.split(":")[1].split("-")[0]) for name in names]
    adata.var["end"] = [int(name.split(":")[1].split("-")[1]) for name in names]
    adata.var["peak_name"] = names
    return adata


def write_peak_bed(adata, path):
    adata.var[["chr", "start", "end", "peak_name"]].to_csv(
        path, sep="\t", header=False, index=False
    )


def read_tf_names(motif_path):
    """Collect the TF names that have at least one motif in the MEME file."""
    tf_names = set()
    with open(motif_path) as handle:
        for line in handle:
            if not line.startswith("MOTIF"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                tf_names.add(parts[2].split("_")[0].strip("()").strip())
    return sorted(tf_names)


def build_promoter_bed(gtf_df, out_path):
    """Write a strand-aware promoter BED file for every gene in the GTF."""
    genes = gtf_df[gtf_df.feature == "gene"].drop_duplicates("gene_name").copy()

    is_plus = genes["strand"] == "+"
    tss = np.where(is_plus, genes["start"], genes["end"])

    genes["prom_start"] = np.where(
        is_plus, tss - PROMOTER_UPSTREAM, tss - PROMOTER_DOWNSTREAM
    ).clip(min=0)
    genes["prom_end"] = np.where(
        is_plus, tss + PROMOTER_DOWNSTREAM, tss + PROMOTER_UPSTREAM
    )

    promoters = genes[["seqname", "prom_start", "prom_end", "gene_name"]]
    promoters.to_csv(out_path, sep="\t", header=False, index=False)
    logger.info(f"Wrote {len(promoters)} promoter regions to {out_path}")


def main():
    logging.getLogger().setLevel(ppConfig.logging_level)
    logger.info("Starting multiome preprocessing")

    # Make sure bedtools from the active environment is found first.
    env_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

    # Directories
    data_dir = Path(ppConfig.data_dir)
    genome_dir = Path(ppConfig.genome_dir)
    motif_dir = Path(ppConfig.motif_directory)
    out_dir = data_dir / ppConfig.output_subdir_name

    for directory in (genome_dir, motif_dir, out_dir):
        create_dir_if_not_exists(directory)

    # Reference genome, annotation and chromosome sizes
    download_genome_references(
        genome_dir=genome_dir,
        species=ppConfig.species,
        assembly=ppConfig.genome_assembly,
        gtf_url=ppConfig.gtf_url,
        chrom_sizes_url=ppConfig.chrom_sizes_url,
        fasta_url=ppConfig.fasta_url,
    )

    # Load both modalities and keep cells present in each
    data_rna, data_atac = load_anndata(
        data_dir, ppConfig.rna_adata_file_name, ppConfig.atac_adata_file_name
    )
    data_rna, data_atac = intersect_cells(data_rna, data_atac)
    data_rna = remove_mitochondrial_genes(
        data_rna, mito_prefix=ppConfig.mitochondrial_prefix
    )

    # Drop cells that failed QC, then keep ATAC in sync
    data_rna = data_rna[data_rna.obs["qc_flag_final"] != "Red", :]
    data_atac = data_atac[data_rna.obs_names, :].copy()
    logger.info(f"After QC filtering: {data_rna.n_obs} cells")

    # Keep protein coding genes only
    gtf_df = load_gtf(genome_dir / "annotation.gtf")
    data_rna = filter_protein_coding_genes(data_rna, gtf_df)

    # Highly variable genes and TFs. User supplied genes and TFs are always
    # kept; the rest are filled up with the most variable ones. Only TFs with
    # a motif in the database are considered.
    motif_path = motif_dir / f"{ppConfig.motif_database}_{ppConfig.species}.meme"
    tf_names_all = read_tf_names(motif_path)

    data_rna, final_genes, final_tfs = compute_hvgs_and_tfs(
        data_rna=data_rna,
        tf_names=tf_names_all,
        user_genes=ppConfig.genes_user,
        user_tfs=ppConfig.tfs_user,
        num_genes=ppConfig.num_genes,
        num_tfs=ppConfig.num_tfs,
        min_cells=ppConfig.min_cells_per_gene,
    )

    # Extend each selected gene (TFs included) by the configured window
    chrom_sizes_path = genome_dir / f"{ppConfig.genome_assembly}.chrom.sizes"
    extended_genes = create_extended_gene_bed(
        gtf_df,
        final_genes + final_tfs,
        window_size=ppConfig.window_size,
        chrom_sizes_path=chrom_sizes_path,
    )
    gene_bed_file = out_dir / f"genes_extended_{ppConfig.window_size // 1000}kb.bed"
    extended_genes.to_csv(gene_bed_file, sep="\t", header=False, index=False)
    logger.info(f"Wrote extended gene windows to {gene_bed_file}")

    # BED file of all peaks
    data_atac = add_peak_coordinates(data_atac)
    all_peaks_bed = out_dir / "peaks_all.bed"
    write_peak_bed(data_atac, all_peaks_bed)

    # Keep only peaks that fall within the window of at least one selected gene
    intersected_bed = out_dir / "peaks_intersected.bed"
    run_bedtools_intersect(
        a_bed=all_peaks_bed, b_bed=gene_bed_file, out_bed=intersected_bed
    )
    peaks_near_genes = pd.read_csv(intersected_bed, sep="\t", header=None)[3]
    data_atac = data_atac[:, list(set(peaks_near_genes))].copy()
    logger.info(f"Peaks near selected genes: {data_atac.n_vars}")

    # Metacells from fine grained Leiden clustering on RNA. These are used
    # for highly variable peak selection and the in silico ChIP-seq step.
    rna_metacell, atac_metacell = create_metacells(
        data_rna,
        data_atac,
        grouping_key="leiden",
        resolution=ppConfig.leiden_resolution,
        batch_key=ppConfig.batch_key,
    )
    data_atac.obs["leiden"] = data_rna.obs["leiden"]

    # Promoters are rebuilt from the current annotation and intersected with
    # this run's own peaks, so no stale promoter file from an older peak set
    # gets reused.
    promoter_bed_file = out_dir / "promoters.bed"
    build_promoter_bed(gtf_df, promoter_bed_file)

    promoter_peaks_bed = out_dir / "promoter_peaks.bed"
    run_bedtools_intersect(
        a_bed=all_peaks_bed, b_bed=promoter_bed_file, out_bed=promoter_peaks_bed
    )
    promoter_hits = pd.read_csv(promoter_peaks_bed, sep="\t", header=None)[3]
    data_atac.var["promoter_col"] = data_atac.var_names.isin(promoter_hits)
    logger.info(f"Promoter peaks found: {data_atac.var['promoter_col'].sum()}")

    # Keep all promoter peaks and fill up to num_peaks with highly variable ones
    data_atac = keep_promoters_and_select_hv_peaks(
        data_atac=data_atac,
        total_n_peaks=ppConfig.num_peaks,
        cluster_key="leiden",
        promoter_col=ppConfig.promoter_col,
    )
    logger.info(f"Final ATAC shape: {data_atac.shape}")

    save_processed_datasets(data_rna, data_atac, out_dir)

    # BED file of the final peak set
    data_atac = add_peak_coordinates(data_atac)
    peaks_bed = out_dir / "peaks_selected.bed"
    write_peak_bed(data_atac, peaks_bed)

    # Motif scores for the selected peaks and TFs
    pwms_sub, key_to_tf = load_motif_database(motif_path, final_tfs)
    fasta_path = genome_dir / f"{ppConfig.genome_assembly}.fa"
    motif_scores = compute_motif_scores(
        bed_file=peaks_bed,
        fasta_file=fasta_path,
        pwms_sub=pwms_sub,
        key_to_tf=key_to_tf,
        n_peaks=data_atac.n_vars,
        window=500,
        threshold=ppConfig.motif_match_pvalue_threshold,
    )
    motif_scores = motif_scores[final_tfs]
    motif_scores.to_csv(out_dir / "motif_scores.tsv", sep="\t")

    # In silico ChIP-seq: correlate TF expression with peak accessibility
    # across metacells, threshold against a background of non motif peaks,
    # then weight by motif scores. Adapted from the approach in
    # https://www.biorxiv.org/content/10.1101/2022.06.15.496239v1 and
    # diffTF (https://pubmed.ncbi.nlm.nih.gov/31801079/).
    atac_metacell = atac_metacell[:, data_atac.var_names].copy()
    tf_mask = rna_metacell.var["gene_type"] == "TF"
    rna_matrix = rna_metacell.X[:, tf_mask]
    atac_matrix = atac_metacell.X

    chipseq_act, chipseq_rep = compute_in_silico_chipseq(
        atac_matrix=atac_matrix,
        rna_matrix=rna_matrix,
        motif_scores=motif_scores,
        percentile=ppConfig.correlation_percentile,
        n_bg=ppConfig.n_bg_peaks_for_corr,
    )
    np.save(out_dir / "insilico_chipseq_act.npy", chipseq_act)
    np.save(out_dir / "insilico_chipseq_rep.npy", chipseq_rep)

    # Gene-peak distances. A distance of 0 means the peak midpoint lies in the
    # gene body or promoter; -1 means the peak is on a different chromosome.
    data_atac.var["index_int"] = range(data_atac.n_vars)

    gene_info = gtf_df[gtf_df.feature == "gene"].drop_duplicates("gene_name").copy()
    gene_info["gene"] = gene_info["gene_name"].values
    gene_info = gene_info.set_index("gene_name")
    gene_info = gene_info.loc[data_rna.var_names.intersection(gene_info.index)]
    gene_info["chr"] = gene_info["seqname"]
    gene_info = gene_info[["chr", "start", "end", "strand", "gene"]].copy()
    gene_info.columns = ["chr_gene", "start", "end", "strand", "gene"]

    dist_matrix = compute_gene_peak_distance_matrix(
        data_rna=data_rna, data_atac=data_atac, gene_coordinates_intersect=gene_info
    )
    np.save(out_dir / "gene_peak_distance_raw.npy", dist_matrix)

    # Exponential distance decay used to initialise the peak-gene matrix.
    # Peaks on other chromosomes get a huge distance so they decay to 0.
    dist_matrix[dist_matrix < 0] = 1e8
    decay = np.exp(-dist_matrix.astype(float) / ppConfig.peak_distance_scaling_factor)
    decay = np.where(decay < ppConfig.peak_distance_min_cutoff, 0, decay)
    np.save(out_dir / "gene_peak_distance_exp.npy", decay)

    logger.info("Preprocessing finished")


if __name__ == "__main__":
    main()
