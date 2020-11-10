# %% [markdown]
# ## We benchmark the neural network to the same dataset that we used for our experiments 


# %% [markdown]
# ## Load neural network
# %%
from eq_learner.architectures.naive_neural_network import ThreeLayerNet
models = ThreeLayerNet(1, 1000, 1).cuda().float()
#


# %% [markdown]
# ## Load datasets
# %%
import torch
from eq_learner.evaluation import eqs_dataset_finder
seen_eq = torch.load('data/1000_train.pt')
novel_eq = torch.load('data/1000_val.pt')

seen_train, seen_val = eqs_dataset_finder.convert_dataset_to_neural_network(seen_eq)
novel_train, novel_val = eqs_dataset_finder.convert_dataset_to_neural_network(novel_eq)