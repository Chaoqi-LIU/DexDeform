import os 
from pathlib import Path
import numpy as np
import tqdm
import yaml
import torch
from pytorch3d.ops import sample_farthest_points

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
goal_particles = goal_state[0] # (10000, 3)
# env.simulator.set_state(0, goal_state)
# img = vis.render()
# plt.imsave("imgs/demo.png", img)

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

mse_loss = torch.nn.MSELoss()

# forward dynamics
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

particle_types = np.zeros((particles.shape[0]))
particle_types[-19:] = 1
particle_types = torch.from_numpy(particle_types).long().to(device) # (10019,)
particle_types = particle_types.unsqueeze(0) # (1, 10019)

input_particles = torch.from_numpy(particles).float().to(device) # (10019, 3)
input_particles = input_particles[None, :, None, :] # (1, 10019, 1, 3)
input_particles = input_particles.repeat(1, 1, num_history, 1) # (1, 10019, 4, 3)

# set action
n_hands = 1
act_dim = len(env.simulator.torch_action_scale)
def rand_action(n_hands=1):
    return np.random.random((n_hands, act_dim))

particles_list = [particles]
for i in range(50):
    action = rand_action(n_hands)
    env.simulator.step(action)
    img = vis.render()
    plt.imsave(f"imgs/steps/{i}.png", img)

    step_state = env.simulator.get_state(0)
    gt_scene_points = np.asarray(step_state[0]) # (10000, 3)
    gt_scene_points = gt_scene_points[downsample_indices]
    step_hand_points = np.stack(step_state[4:-2])[:, :3] # (19, 3)

    input_actions = torch.zeros((1, input_particles.size(1), 3)).float().to(device) # (1, 10019, 3)
    input_actions[:, -19:, :] = torch.from_numpy(step_hand_points - hand_points).float().to(device) # (1, 10019, 3)

    pred_particles = model(input_particles, input_actions, particle_types) # (1, 10019, 3)

    pred = pred_particles[0, :num_points, :] # (10000, 3)
    gt = torch.from_numpy(gt_scene_points).float().to(device) # (10000, 3)
    loss = mse_loss(pred, gt)
    print(f'step {i}, loss: {loss.item()}')
    
    # plot
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    
    hand_edges = []
    for edge in RIGHT_HAND_EDGES:
        start = step_hand_points[edge[0]]
        end = step_hand_points[edge[1]]                
        # (x, y, z) -> (z, x, y)
        start = [start[2], start[0], start[1]]
        end = [end[2], end[0], end[1]]
        hand_edges.append([start, end])
    
    fig = plt.figure(figsize=(10, 16))
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(pred[:, 2], pred[:, 0], pred[:, 1], c='r', marker='o')
    ax1.scatter(step_hand_points[:, 2], step_hand_points[:, 0], step_hand_points[:, 1], c='g', marker='x')
    
    # Add edges as lines
    hand_lines = Line3DCollection(hand_edges, colors='g', linewidths=1.)
    ax1.add_collection3d(hand_lines)
    
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax1.set_aspect('equal')
    ax1.legend(['predicted', 'hand'])
    
    ax2 = fig.add_subplot(212, projection='3d')
    # ax.scatter(obj_pred_particles[:, 2], obj_pred_particles[:, 0], obj_pred_particles[:, 1], c='r', marker='o')
    ax2.scatter(gt[:, 2], gt[:, 0], gt[:, 1], c='b', marker='o')
    ax2.scatter(step_hand_points[:, 2], step_hand_points[:, 0], step_hand_points[:, 1], c='g', marker='x')
    
    # Add edges as lines
    hand_lines = Line3DCollection(hand_edges, colors='g', linewidths=1.)
    ax2.add_collection3d(hand_lines)
    
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')
    ax2.set_aspect('equal')
    ax2.legend(['gt', 'hand'])

    plt.tight_layout()
    plt.savefig(f"imgs/pred/{i}.png")
    plt.close()
    
    if i < 10:
        # update input_particles
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
