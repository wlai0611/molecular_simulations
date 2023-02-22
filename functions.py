import torch

def interatomic_xyz_distances(coordinates):
    return torch.unsqueeze(coordinates, dim=1) - coordinates  
