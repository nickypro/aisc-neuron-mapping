import torch
import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from k_means_constrained import KMeansConstrained
from taker import Model


def cluster_neurons(model: Model,
        layer: int,
        split_num: int=96,
        method="kmeans",
    ):
    # First, get variables for which components are used
    assert model.cfg.d_mlp % split_num == 0, \
        "split_num should evenly divide model's mlp width"
    split_size = model.cfg.d_mlp // split_num

    # Collect the neurons we are clustering
    weights = model.layers[layer]["mlp.W_in"].detach().cpu()
    normed_weights = sk_normalize(weights)

    # Perform the clustering
    if method == "kmeans":
        kmeans = KMeansConstrained(
            n_clusters=split_num, size_min=split_size, size_max=split_size, random_state=0
        ).fit(normed_weights, None)
        labels = [x for x in kmeans.labels_]
        return labels, kmeans

    if method == "random":
         labels = np.array(list(range(model.cfg.d_mlp))) % split_num
         return labels, {}

    raise NotImplementedError(f"method {method} not implemented")
