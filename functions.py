import torch
import re
import time

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

def get_delta_velocity(forces, masses, delta_time):
    '''
    a = F/m = dv/dt
    dv = a * dt
    '''
    return torch.matmul(torch.diag(1/masses), forces) * delta_time/2
    
def save_trajectory(positions, trajectory_file):
    m,n        = positions.shape
    outstring  = str(m)+"\n\n"
    for coordinates in positions:
      outstring += "LJ "+" ".join([str(coordinate.item()) for coordinate in coordinates])+"\n"
    trajectory_file.write(outstring)

def load_tensor_from_xyz(filename):
    file     = open(filename, mode='r')
    contents = file.read()
    data = []
    lines= contents.split('\n')
    for line in lines:
      if line.startswith('LJ'):
        vals = re.split('\s+',line)[1:]
        data.append([float(val) for val in vals])
    return torch.tensor(data)

def get_trajectories(coordinates, mass=1, timesteps=10000, delta_time=0.001):
    num_atoms, num_dimensions = coordinates.shape
    velocities = torch.zeros(num_atoms, num_dimensions)
    masses     = torch.ones(num_atoms)*mass

    timestamp  = str(round(time.time()))
    filename   = f"trajectory{timestamp}.xyz"
    file       = open(file=filename, mode='a')
    
    xyz_distances = interatomic_xyz_distances(coordinates)
    xyz_forces    = interatomic_xyz_forces(xyz_distances)

    for timestep in range(timesteps):
      #predictor
      delta_velocities = get_delta_velocity(xyz_forces, masses, delta_time)      
      velocities       +=delta_velocities
      coordinates      +=velocities * delta_time
      xyz_distances    = interatomic_xyz_distances(coordinates)
      xyz_forces       = interatomic_xyz_forces(xyz_distances)
      #corrector
      delta_velocities = get_delta_velocity(xyz_forces, masses, delta_time)
      velocities       +=delta_velocities
      save_trajectory(coordinates, file)
    return filename

def potential_return(r_t,epsilon,sigma_t):
    z = sigma_t/r_t
    u = z*z*z
    return -4*epsilon*u*(1-u)

def compute_potential_energy(distances):
    potentials_between_atoms = potential_return(distances,1,1)
    return torch.sum(torch.triu(potentials_between_atoms,diagonal = 1)).item()
