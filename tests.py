import torch
import functions
import pickle
#import pdb;pdb.set_trace()

coordinates = torch.tensor([
  [0.5391356726,0.1106588251,-0.4635601962],
  [-0.5185079933,0.4850176090,0.0537084789],
  [0.0793723207,-0.4956764341,0.5098517173],
  ])
observed_distances = functions.interatomic_xyz_distances(coordinates)
  

try:
  
  true_trajectory = torch.tensor([

    [[0, 0, 0],
    [1.0576, -0.37436, -0.5173],
    [0.459763,0.606335,-0.9734]],

    [[-1.0576, 0.37436, 0.5173],
    [0.00005,0,0],
    [-0.59788,0.98069,-0.45614]],

   [[-0.459763,-0.606335,0.9734],
   [0.59788,-0.98069,0.45614],
   [0,0,0]],

  ])
  torch.testing.assert_close(true_trajectory,observed_distances,rtol=1e-04,atol=1e-04)
except:
  print('interatomic distances function broke')
  print(observed_distances)


observed_forces = functions.interatomic_xyz_forces(observed_distances)

try:
  true_forces = torch.tensor([
        [-2.9359, -0.4488,  2.8842],
        [ 3.2031, -2.6216, -0.1184],
        [-0.2671,  3.0704, -2.7658]])
  torch.testing.assert_close(true_forces,observed_forces,rtol=1e-04,atol=1e-04)
except:
  print('force calculation function wrong')
  print(observed_forces)

pickle.dump(observed_forces, open("forces.pickle",'wb'))

masses = torch.tensor([1,1,1])
observed_delta_velocity = functions.get_delta_velocity(forces=observed_forces, masses=masses, delta_time=0.001)
try:
  actual_delta_velocity = torch.tensor(
  [[-1.46795825e-03, -2.24405332e-04,  1.44209151e-03],
  [ 1.60152952e-03, -1.31081245e-03, -5.91763254e-05],
  [-1.33571269e-04,  1.53521779e-03, -1.38291519e-03],]
  )  
  torch.testing.assert_close(actual_delta_velocity, observed_delta_velocity,rtol=1e-04,atol=1e-04)
except:
  print('get delta velocity broke')
  print(observed_delta_velocity)

outfilename = functions.get_trajectories(coordinates)
observed_trajectory = functions.load_tensor_from_xyz(outfilename)
try:
  actual_trajectory = functions.load_tensor_from_xyz('trajectory.xyz')
  torch.testing.assert_close(actual_trajectory,observed_trajectory,rtol=1e-01,atol=1e-01)
except:
  print('get trajectories function broke')
  print(observed_trajectory[:10,:])


#LJ        0.539134        0.110659       -0.463559
#LJ       -0.518506        0.485016        0.053708
#LJ        0.079372       -0.495675        0.509850
#[[ 0.53913567  0.11065883 -0.4635602 ]
# [-0.51850799  0.48501761  0.05370848]
# [ 0.07937232 -0.49567643  0.50985172]]

actual_potential_energy = -2.424541
distances = torch.sum(observed_distances**2,dim=2)
try:
  observed_potential_energy = functions.compute_potential_energy(distances)
  torch.testing.assert_close(actual_potential_energy, observed_potential_energy, rtol=1e-04, atol=1e-04)
except Exception as e:
  print('potential energy function broke')
  print(observed_potential_energy)
  print(e.args)
