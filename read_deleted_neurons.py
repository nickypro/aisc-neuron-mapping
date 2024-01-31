import torch
import os
import matplotlib.pyplot as plt 

torch.set_printoptions(threshold=10000)

def load_last_n_files(n: int):
    saved_deleted_neurons = os.listdir("tmp/15M")[-n:]

    return [torch.load("tmp/15M/" + filename) for filename in saved_deleted_neurons]
        

history = torch.load("tmp/15M/history/history.pt")

plt.imshow(load_last_n_files(1)[0]["ff_criteria"])
plt.show()