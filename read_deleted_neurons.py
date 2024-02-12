import torch
import os
import matplotlib.pyplot as plt

from taker.data_classes import RunDataHistory 

torch.set_printoptions(threshold=10000)

def load_last_n_files(n: int, path: str = "tmp/15M"):
    saved_deleted_neurons = os.listdir(path)[-n:]

    return [torch.load(f"{path}{filename}") for filename in saved_deleted_neurons]

def load_results_for(cripple_datasets: list[str], focus_dataset: str, model_size: str, fraction: str):
    return [torch.load(f"examples/neuron-mapping/saved_tensors/{model_size}/{cripple_dataset}-{focus_dataset}-{model_size}-{fraction}-recent.pt") for cripple_dataset in cripple_datasets]      

def get_performance_history(run_data_history: RunDataHistory, focus: str, cripple: str):
    return run_data_history[f"{focus}_loss"], run_data_history[f"{cripple}_loss"]

cifar20_datasets = [f"cifar20-{dataset}" for dataset in ["aquatic_mammals", "fish", "flowers", "food_containers",
                    "fruit_and_vegetables", "household_electrical_devices", "household_furniture", "insects",
                    "large_carnivores", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
                    "medium_sized_mammals", "non_insect_invertebrates", "people", "reptiles", "small_mammals",
                    "trees", "veh1", "veh2"]]

results = load_results_for(
        cripple_datasets=cifar20_datasets,
        focus_dataset="cifar20-split",
        model_size="Cifar100",
        fraction="0.01"
)

print(results)