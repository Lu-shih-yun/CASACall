import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data

def dataset(dataset:str):

    if dataset == "bonito_training_data":
        input = torch.from_numpy(np.load("preprocess/bonito_training_data/Chunk.npy")).float()
        target = torch.from_numpy(np.load("preprocess/bonito_training_data/Reference.npy"))
        target_lengths = torch.from_numpy(np.int16(np.load("preprocess/bonito_training_data/Reference_length.npy")))
        valid_lengths = torch.from_numpy(np.int16(np.load("preprocess/bonito_training_data/validation/Reference_length.npy")))
        val_input = torch.from_numpy(np.load("preprocess/bonito_training_data/validation/Chunk.npy")).float()
        val_target = torch.from_numpy(np.load("preprocess/bonito_training_data/validation/Reference.npy"))

    torch_dataset = data.TensorDataset(input,target,target_lengths)
    val_dataset = data.TensorDataset(val_input,val_target,valid_lengths)
    return torch_dataset,val_dataset



