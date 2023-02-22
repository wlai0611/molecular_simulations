import torch

def interatomic_xyz_distances(coordinates):
    rows, columns = coordinates.shape
    return torch.reshape(coordinates, shape=(rows, 1, columns)) - coordinates  

def interatomic_xyz_forces(coordinates):
    return torch.tensor([])
