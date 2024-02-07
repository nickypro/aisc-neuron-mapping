from taker.data_classes import PruningConfig
from taker.prune import run_pruning

import torch

pruningConfig = PruningConfig(
    model_repo="Ahmed9275/Vit-Cifar100",
    token_limit  = 1000,  # trim the input to this max length
    run_pre_test = True,  # evaluate the unpruned model
    eval_sample_size = 1e2,
    collection_sample_size = 1e2,
    # Removals parameters
    ff_frac   = 0.2,     # % of feed forward neurons to prune
    attn_frac = 0.00,     # % of attention neurons to prune
    focus     = "cifar20-split", # the “reference” dataset
    cripple   = "cifar20-veh1",          # the “unlearned” dataset
    additional_datasets=tuple(), # any extra datasets to evaluate on
    recalculate_activations = False, # iterative vs non-iterative
    n_steps = 1,
    wandb_project = "testing-tetra-image"
)

with torch.no_grad():
    model, history = run_pruning(pruningConfig)
