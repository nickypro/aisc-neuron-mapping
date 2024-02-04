from taker.data_classes import PruningConfig
from taker.prune import run_pruning

import torch

pruningConfig = PruningConfig(
    model_repo="Ahmed9275/Vit-Cifar100",
    run_pre_test=False,  # evaluate the unpruned model

    ff_frac=0.1,  # % of feed forward neurons to prune
    attn_frac=0.00,  # % of attention neurons to prune
    focus="cifar20-split",  # the “reference” dataset
    cripple="cifar20-veh1",  # the “unlearned” dataset
    additional_datasets=tuple(),  # any extra datasets to evaluate on
    recalculate_activations=False,  # iterative vs non-iterative
    eval_sample_size=100,  # number of tokens to evaluate on
    collection_sample_size=100,  # number of tokens to collect activations on
    save=True,
    n_steps=20
)

with torch.no_grad():
    model, history = run_pruning(pruningConfig)
