import torch

def interatomic_xyz_distances(coordinates):
    rows, columns = coordinates.shape
    return torch.reshape(coordinates, shape=(rows, 1, columns)) - coordinates  

def gradient_return(r_t,epsilon,sigma_t):
    z = sigma_t/r_t
    u = z*z*z
    return (24*epsilon*u*(1-2*u)/r_t)

def interatomic_xyz_forces(xyz_distances):
    squared_euclidean_distances = torch.sum(xyz_distances**2, dim=2)
    rows, columns = squared_euclidean_distances.shape
    gradients = gradient_return(r_t=squared_euclidean_distances+torch.eye(rows,rows),epsilon=1,sigma_t=1) #adding torch.eye avoids division by 0
    gradients = gradients - torch.diag(torch.diag(gradients))                     #make the diagonal 0s again
    return torch.sum(xyz_distances * torch.reshape(gradients,shape=(rows,columns,1)),dim=0)    
