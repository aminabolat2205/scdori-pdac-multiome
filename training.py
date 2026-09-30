#!/usr/bin/env python
"""Train an scDoRI model on preprocessed multiome data.

Training runs in two phases:

    Phase 1: learn topics by reconstructing ATAC peaks (module 1), RNA from
             predicted accessibility (module 2) and TF expression (module 3).
             A warmup trains modules 1 and 3 first before module 2 is added.
    Phase 2: starting from the best Phase 1 checkpoint, learn activator and
             repressor TF-gene links per topic (module 4). Earlier modules
             can optionally be frozen for stability.

All settings are read from scdori.trainConfig.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from scdori import (
    trainConfig,
    initialize_scdori_parameters,
    load_best_model,
    load_scdori_inputs,
    save_model_weights,
    scDoRI,
    set_seed,
    train_model_grn,
    train_scdori_phases,
)

logger = logging.getLogger(__name__)

EVAL_FRACTION = 0.2
SPLIT_SEED = 42


def make_loader(indices, shuffle):
    dataset = TensorDataset(torch.from_numpy(indices))
    return DataLoader(dataset, batch_size=trainConfig.batch_size_cell, shuffle=shuffle)


def main():
    logging.basicConfig(level=trainConfig.logging_level)
    set_seed(trainConfig.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting scDoRI training on {device}")

    # Processed RNA/ATAC data, in silico ChIP-seq matrices and gene-peak
    # distances, all loaded from the paths in the config.
    rna_metacell, atac_metacell, gene_peak_dist, insilico_act, insilico_rep = (
        load_scdori_inputs(trainConfig)
    )

    # Binary mask of which peak-gene links are allowed, based on distance
    gene_peak_fixed = gene_peak_dist.clone()
    gene_peak_fixed[gene_peak_fixed > 0] = 1

    # Each observation counts as one cell (this would be >1 for real metacells)
    rna_metacell.obs["num_cells"] = 1
    num_cells = rna_metacell.obs["num_cells"].values.reshape(-1, 1)

    # Positions of the TFs within the gene axis
    rna_metacell.var["index_int"] = range(rna_metacell.n_vars)
    tf_indices = rna_metacell.var.loc[
        rna_metacell.var["gene_type"] == "TF", "index_int"
    ].values

    # One-hot encode the technical batch. The ATAC object gets its batch
    # labels from RNA so both modalities are guaranteed to match.
    batch_labels = rna_metacell.obs[trainConfig.batch_col].values
    rna_metacell.obs["batch"] = batch_labels
    atac_metacell.obs["batch"] = batch_labels

    encoder = OneHotEncoder(handle_unknown="ignore")
    onehot_batch = encoder.fit_transform(batch_labels.reshape(-1, 1)).toarray()
    logger.info(f"Batches: {list(encoder.categories_[0])}")

    # Train/eval split over cells
    all_idx = np.arange(rna_metacell.n_obs)
    train_idx, eval_idx = train_test_split(
        all_idx, test_size=EVAL_FRACTION, random_state=SPLIT_SEED
    )
    train_loader = make_loader(train_idx, shuffle=True)
    eval_loader = make_loader(eval_idx, shuffle=False)
    logger.info(f"Training on {len(train_idx)} cells, evaluating on {len(eval_idx)}")

    model = scDoRI(
        device=device,
        num_genes=rna_metacell.n_vars,
        num_peaks=atac_metacell.n_vars,
        num_tfs=insilico_act.shape[1],
        num_topics=trainConfig.num_topics,
        num_batches=onehot_batch.shape[1],
        dim_encoder1=trainConfig.dim_encoder1,
        dim_encoder2=trainConfig.dim_encoder2,
    ).to(device)

    gene_peak_dist = gene_peak_dist.to(device)
    gene_peak_fixed = gene_peak_fixed.to(device)
    insilico_act = insilico_act.to(device)
    insilico_rep = insilico_rep.to(device)

    training_inputs = (
        device,
        train_loader,
        eval_loader,
        rna_metacell,
        atac_metacell,
        num_cells,
        tf_indices,
        onehot_batch,
        trainConfig,
    )

    # Phase 1: initialise from the in silico ChIP-seq scores and distance
    # based peak-gene links. TF-gene links stay frozen during this phase.
    initialize_scdori_parameters(
        model,
        gene_peak_dist,
        gene_peak_fixed,
        insilico_act=insilico_act,
        insilico_rep=insilico_rep,
        phase="warmup",
    )
    model = train_scdori_phases(model, *training_inputs)

    phase1_dir = Path(trainConfig.weights_folder_scdori)
    save_model_weights(model, phase1_dir, "scdori_final")

    # Phase 2: continue from the best Phase 1 checkpoint (not the last epoch)
    # and let the TF-gene links train.
    model = load_best_model(model, phase1_dir / "best_scdori_best_eval.pth", device)
    initialize_scdori_parameters(
        model,
        gene_peak_dist,
        gene_peak_fixed,
        insilico_act=insilico_act,
        insilico_rep=insilico_rep,
        phase="grn",
    )
    model = train_model_grn(model, *training_inputs)

    save_model_weights(model, Path(trainConfig.weights_folder_grn), "scdori_final")
    logger.info("Training finished")


if __name__ == "__main__":
    main()
