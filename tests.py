import torch
import functions
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
