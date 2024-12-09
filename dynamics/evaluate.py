import os
import numpy as np
import argparse
import time
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch3d.ops import sample_farthest_points

import sys
sys.path.append('.')
from dynamics.dy_utils import *
from dynamics.gnn import GraphNeuralDynamics, NUM_RIGHT_HAND_PARTICLES, RIGHT_HAND_EDGES
from dynamics.dataset import GNNDataset




def rollout(config):
    ## config
    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    train_config = config["train_config"]
    rollout_config = config["rollout_config"]

    # misc
    num_history = dataset_config['n_his']
    num_future = dataset_config['n_future']
    
    data_name = dataset_config["data_name"]
    rollout_out_dir = os.path.join(rollout_config["out_dir"], data_name, '0002')
    os.makedirs(rollout_out_dir, exist_ok=True)
    
    torch.autograd.set_detect_anomaly(True)
    set_seed(train_config['random_seed'])
    device = torch.device(dataset_config['device'])
    
    ## load dataset
    phase = 'valid'
    dataset = GNNDataset(
        dataset_config=dataset_config,
        phase=phase
    )
    
    dataloaders = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers']
    ) 
    dataloaders = dataloader_wrapper(dataloaders, phase)
    
    ## load model
    model = GraphNeuralDynamics(
        num_prop_steps=3,
        use_velocity=False,
        latent_dim=model_config['latent_dim'],
        history_window_size=num_history,
        connectivity_radius=model_config['cradius'],
        max_num_neighbors=model_config['nngbrs'],
    )
    model.to(device)
    
    checkpoint_dir = os.path.join(train_config['out_dir'], data_name)
    checkpoint_path = os.path.join(checkpoint_dir, 'best.pth')
    
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path))
    
    mse_loss = torch.nn.MSELoss()
    
    ## rollout
    total_loss = []
    for i in range(len(dataset)):
        data = dataset[i]
        particles = data['particles'].unsqueeze(0).to(device)               # (B, N, T, 3)
        actions = data['actions'].unsqueeze(0).to(device)                 # (B, N, T-1, 3)
        particle_types = data['particle_types'].unsqueeze(0).to(device)     # (B, N, T)
        B, N, T, _ = particles.size()
        
        # # sample the initial particles
        # batch_idx = torch.arange(particles.size(0)).view(-1, 1).expand(-1, dataset_config['n_obj_ptcls'])
        # _, scene_downsample_idx = sample_farthest_points(
        #     particles[particle_types == 0, :, 0].reshape(B, -1, 3),
        #     K=dataset_config['n_obj_ptcls'],
        #     random_start_point=True
        # )
        
        # downsample_mask = torch.zeros_like(particle_types)
        # downsample_mask[particle_types == 1] = 1
        # downsample_mask[batch_idx, scene_downsample_idx] = 1
        # downsample_mask = downsample_mask.bool()

        # particles = particles[downsample_mask].reshape(B, -1, T, 3).to(device)       # (B, N_sample, T, 3)
        # actions = actions[downsample_mask].reshape(B, -1, T-1, 3).to(device)         # (B, N_sample, T-1, 3)  
        # particle_types = particle_types[downsample_mask].reshape(B, -1).to(device)   # (B, N_sample)
        
        loss = 0
        for fi in range(dataset_config['n_future']):
            this_particles = particles[:, :, fi:fi+num_history].clone()
            assert this_particles.size(2) == num_history
            this_actions = actions[:, :, num_history+fi-1].clone()
            this_pred_particles = model(
                this_particles,
                this_actions,
                particle_types,
            )       # (B, N, 3)
            this_gt_particles = particles[:, :, num_history+fi]
            this_loss = mse_loss(this_pred_particles, this_gt_particles)
            loss += this_loss
            
            # plot the predicted particles and the ground truth particles
            this_pred_particles = this_pred_particles[0].cpu().detach().numpy() # (N, 3)
            this_gt_particles = this_gt_particles[0].cpu().detach().numpy()     # (N, 3)
            
            obj_pred_particles = this_pred_particles[:-NUM_RIGHT_HAND_PARTICLES]
            obj_gt_particles = this_gt_particles[:-NUM_RIGHT_HAND_PARTICLES]
            
            hand_particles = this_pred_particles[-NUM_RIGHT_HAND_PARTICLES:] # (19, 3)
            
            # Create hand edges
            hand_edges = []
            for edge in RIGHT_HAND_EDGES:
                start = hand_particles[edge[0]]
                end = hand_particles[edge[1]]                
                # (x, y, z) -> (z, x, y)
                start = [start[2], start[0], start[1]]
                end = [end[2], end[0], end[1]]
                hand_edges.append([start, end])
            
            # plot
            fig = plt.figure(figsize=(10, 16))
            ax1 = fig.add_subplot(211, projection='3d')
            ax1.scatter(obj_pred_particles[:, 2], obj_pred_particles[:, 0], obj_pred_particles[:, 1], c='r', marker='o')
            # ax.scatter(obj_gt_particles[:, 2], obj_gt_particles[:, 0], obj_gt_particles[:, 1], c='b', marker='x')
            ax1.scatter(hand_particles[:, 2], hand_particles[:, 0], hand_particles[:, 1], c='g', marker='x')
            
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
            ax2.scatter(obj_gt_particles[:, 2], obj_gt_particles[:, 0], obj_gt_particles[:, 1], c='b', marker='o')
            ax2.scatter(hand_particles[:, 2], hand_particles[:, 0], hand_particles[:, 1], c='g', marker='x')
            
            # Add edges as lines
            hand_lines = Line3DCollection(hand_edges, colors='g', linewidths=1.)
            ax2.add_collection3d(hand_lines)
            
            ax2.set_xlabel('X Label')
            ax2.set_ylabel('Y Label')
            ax2.set_zlabel('Z Label')
            ax2.set_aspect('equal')
            ax2.legend(['gt', 'hand'])

            plt.tight_layout()
            plt.savefig(os.path.join(rollout_out_dir, f'{i:04d}_{fi}.png'))
            plt.close()
        
        total_loss.append(loss.item())
    
    avg_total_loss = np.mean(total_loss)
    print(f"Average total loss: {avg_total_loss}")
        
    
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/dexdeform_folding.yaml')
    args = arg_parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    rollout(config)