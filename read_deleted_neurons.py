import torch
import os
import matplotlib.pyplot as plt

from taker.data_classes import RunDataHistory 

torch.set_printoptions(threshold=10000)

def load_last_n_files(n: int):
    saved_deleted_neurons = os.listdir("tmp/15M")[-n:]

    return [torch.load("tmp/15M/" + filename) for filename in saved_deleted_neurons]
        

def get_performance_history(history: RunDataHistory, focus: str, cripple: str):
    return history[f"{focus}_loss"], history[f"{cripple}_loss"]

history: RunDataHistory = torch.load("tmp/15M/history/history.pt")

plt.imshow(load_last_n_files(1)[0]["ff_criteria"])
plt.show()