import torch
import functions
import cProfile

coordinates = torch.rand(500,3)*20

#start = torch.cuda.Event(enable_timing=True)
#end   = torch.cuda.Event(enable_timing=True)
#start.record()

cProfile.run('functions.get_trajectories(coordinates)')
#print(functions.get_trajectories(coordinates))
#end.record()
#torch.cuda.synchronize()
#print(start.elapsed_time(end))

