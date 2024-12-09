import os 
from pathlib import Path
import numpy as np
import tqdm
import yaml
import torch
from torch import nn
from pytorch3d.ops import sample_farthest_points
from geomloss import SamplesLoss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dynamics.gnn import GraphNeuralDynamics
from mpm.hand import HandEnv
from mpm import make
from mpm.viewer import Viewer
from mpm.video_utils import write_video
from policy.utils.io import load_gzip_file

NUM_RIGHT_step_hand_points = 19
RIGHT_HAND_EDGES = np.array([
    [17, 18],[18, 17],[16, 17],[17, 16],[16, 18],[18, 16],
    [ 4,  5],[ 5,  4],[ 3,  4],[ 4,  3],[ 3,  5],[ 5,  3],
    [ 7,  8],[ 8,  7],[ 6,  7],[ 7,  6],[ 6,  8],[ 8,  6],
    [10, 11],[11, 10],[ 9, 10],[10,  9],[ 9, 11],[11,  9],
    [14, 15],[15, 14],[13, 14],[14, 13],[13, 15],[15, 13],
    [16,  3],[ 3, 16],[ 3,  6],[ 6,  3],[ 6,  9],[ 9,  6],
    [ 9, 13],[13,  9],[ 1, 16],[16,  1],[ 1,  3],[ 3,  1],
    [ 1,  6],[ 6,  1],[ 1,  9],[ 9,  1],[ 1, 13],[13,  1],
    [12, 16],[16, 12],[12,  3],[ 3, 12],[12,  6],[ 6, 12],
    [12,  9],[ 9, 12],[12, 13],[13, 12],[ 1, 12],[12,  1],
    [ 1,  2],[ 2,  1],[ 2, 12],[12,  2],[ 0,  1],[ 1,  0],
    [ 0,  2],[ 2,  0],[ 0, 12],[12,  0],[ 0, 16],[16,  0]])

# set up environment and initial state
env = make(env_name='folding', sim_cfg={'max_steps': 4400})
init_state = env.init_state # 0: object point cloud (10000, 3)
init_particles = init_state[0]
tmp_state = init_state

vis = Viewer(env)
vis.refresh_views("obj_centric")
vis.set_view("side")
img = vis.render()
plt.imsave("imgs/init.png", img)

# set up goal state
# read data
demo_path = 'data/folding/0c795a2e62284cf28fffd048ad58b769.pkl'
demo_data = load_gzip_file(demo_path)
demo_states = demo_data['states']
goal_state = demo_states[-1]
goal_particles = np.asarray(goal_state[0]) # (10000, 3)
env.simulator.set_state(0, goal_state)
img = vis.render()
plt.imsave("imgs/target.png", img)

# load model
config_path = 'config/dexdeform_folding.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

model_config = config['model_config']
num_history = 4
device = torch.device('cuda:0')

model = GraphNeuralDynamics(
    num_prop_steps=3,
    use_velocity=False,
    latent_dim=model_config['latent_dim'],
    history_window_size=num_history,
    connectivity_radius=model_config['cradius'],
    max_num_neighbors=model_config['nngbrs'],
)
model.to(device)

model_path = 'ckpts/folding.pth'
model.eval()
model.load_state_dict(torch.load(model_path))

OT_LOSS = SamplesLoss(loss="sinkhorn", p=1, blur=0.0001)

# forward dynamics
env.simulator.set_state(0, init_state)
scene_points = np.asarray(init_state[0]) # (10000, 3)
hand_points = np.stack(init_state[4:-2])[:, :3] # (19, 3)

# downsample
NUM_SCENE_POINTS = 1024
_, downsample_indices = sample_farthest_points(
    torch.from_numpy(scene_points).unsqueeze(0).cuda(),
    K=NUM_SCENE_POINTS, 
    random_start_point=True
)
downsample_indices = downsample_indices.cpu().numpy().squeeze(0)

scene_points = scene_points[downsample_indices]
particles = np.concatenate([scene_points, hand_points], axis=0) # (10019, 3)
num_points = scene_points.shape[0]

goal_points = goal_particles[downsample_indices]

particle_types = np.zeros((particles.shape[0]))
particle_types[-19:] = 1
particle_types = torch.from_numpy(particle_types).long().to(device) # (10019,)
particle_types = particle_types.unsqueeze(0) # (1, 10019)


def forward_dynamics(particles, actions, n_actions=50, viz=False):
    particles_list = [particles]
    
    input_particles = torch.from_numpy(particles).float().to(device) # (10019, 3)
    input_particles = input_particles[None, :, None, :] # (1, 10019, 1, 3)
    input_particles = input_particles.repeat(1, 1, num_history, 1) # (1, 10019, 4, 3)
    
    for i in range(n_actions):
        action = actions[i:i+1]
        env.simulator.step(action)
        
        # get gt state
        step_state = env.simulator.get_state(0)
        gt_scene_points = np.asarray(step_state[0]) # (10000, 3)
        gt_scene_points = gt_scene_points[downsample_indices]
        step_hand_points = np.stack(step_state[4:-2])[:, :3] # (19, 3)
        
        input_actions = torch.zeros((1, input_particles.size(1), 3)).float().to(device) # (1, 10019, 3)
        input_actions[:, -19:, :] = torch.from_numpy(step_hand_points - hand_points).float().to(device) # (1, 10019, 3)

        # forward dynamics
        pred_particles = model(input_particles, input_actions, particle_types) # (1, 10019, 3)
        pred = pred_particles[0, :num_points, :] # (10000, 3)
        
        if i < n_actions-1:
            # update input_particles
            pred = pred.detach().cpu().numpy()
            particles = np.concatenate([pred, step_hand_points], axis=0)
            particles_list.append(particles)
            
            input_particles = np.zeros((particles.shape[0], num_history, 3)) # (10019, 4, 3)
            for j in range(num_history):
                if np.abs(-j-1) > len(particles_list):
                    history_idx = 0
                else:
                    history_idx = -j-1
                input_particles[:, j, :] = particles_list[history_idx]
            input_particles = torch.from_numpy(input_particles).float().to(device)
            input_particles = input_particles[None, :, :, :] # (1, 10019, 4, 3)
    
    return pred
    

actions = nn.Parameter(env.simulator.togpu(np.zeros(shape=(50, 26))))
optim = torch.optim.Adam([actions], lr=1.0)

ran = tqdm.trange(50)
for i in ran:
    optim.zero_grad()
    pred = forward_dynamics(particles, actions, n_actions=50)
    
    # loss = mse_loss(pred, torch.from_numpy(goal_points).float().to(device))
    loss = OT_LOSS(pred, torch.from_numpy(goal_points).float().to(device))
    loss.backward()
    optim.step()
    
    ran.set_description(f'loss: {loss.item()}')

# evaluate
images = []
env.simulator.set_state(0, tmp_state)

for i in tqdm.trange(50):
    env.simulator.step(actions[i:i+1].detach().cpu().numpy())
    images.append(vis.render())

write_video(images, filename="imgs/traj_optim.gif")

# save final image
plt.imsave("imgs/final.png", images[-1])