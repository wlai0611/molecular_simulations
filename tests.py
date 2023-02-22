import torch
import functions

true_trajectory = torch.tensor([
 [0.539134,       0.110659,       -0.463559],
 [-0.518506,      0.485016,        0.053708],
 [0.079372,       -0.495675,       0.509850]
])

actual_trajectory = functions.interatomic_xyz_distances(123)

torch.testing.assert_close(true_trajectory,actual_trajectory,msg='interatomic distances')
