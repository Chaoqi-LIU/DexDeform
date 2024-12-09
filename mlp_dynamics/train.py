import os
import numpy as np
import argparse
import time
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch3d.ops import sample_farthest_points

import sys
sys.path.append('.')
from dynamics.dy_utils import *
from dynamics.dataset import GNNDataset
from mlp_dynamics.pcd_mlp import PointcloudEncoder


def train(config):
    ## config
    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    train_config = config["train_config"]

    # misc
    num_history = dataset_config['n_his']
    num_future = dataset_config['n_future']
    
    data_name = dataset_config["data_name"]
    dyn_out_dir = os.path.join(train_config["out_dir"], data_name)
    os.makedirs(dyn_out_dir, exist_ok=True)
    
    torch.autograd.set_detect_anomaly(True)
    set_seed(train_config['random_seed'])
    device = torch.device(dataset_config['device'])
    
    ## load dataset
    phases = train_config['phases']
    
    datasets = {phase: GNNDataset(
        dataset_config=dataset_config,
        phase=phase
    ) for phase in phases}
    
    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=train_config['batch_size'],
        shuffle=(phase == 'train'),
        num_workers=train_config['num_workers']
    ) for phase in phases}
    
    dataloaders = {phase: dataloader_wrapper(dataloaders[phase], phase) for phase in phases}
    
    ## load model
    model = PointcloudEncoder(
        in_channels=(
            3 * num_history +   # history positions
            3 +                 # action
            1                   # particle type
        ),
        out_channels=3,
    )
    model.to(device)
    
    ## loss function and optimizer
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    
    ## training
    loss_plot_list_train = []
    loss_plot_list_valid = []
    best_valid_error = float('inf')
    for epoch in range(train_config['n_epochs']):
        time1 = time.time()
        for phase in phases:
            model.train(phase == 'train')
            with torch.set_grad_enabled(phase == 'train'):
                
                # iterate over batches
                loss_sum_list = []
                n_iters = train_config['n_iters_per_epoch'][phase] \
                        if train_config['n_iters_per_epoch'][phase] != -1 else len(datasets[phase])
                for i in range(n_iters):
                    data = next(dataloaders[phase])
                    data = {key: value.to(device) for key, value in data.items()}
                    
                    # decouple
                    particles = data['particles']           # (B, N, T, 3)
                    actions = data['actions']               # (B, N, T-1, 3)
                    particle_types = data['particle_types'] # (B, N)
                    B, N, T, _ = particles.size()
                    assert particles.size(2) == num_history + num_future

                    # downsample
                    batch_idx = torch.arange(particles.size(0)).view(-1, 1).expand(-1, dataset_config['n_obj_ptcls'])
                    _, scene_downsample_idx = sample_farthest_points(
                        particles[particle_types == 0, :, 0].reshape(B, -1, 3),
                        K=dataset_config['n_obj_ptcls'],
                        random_start_point=True
                    )
                    downsample_mask = torch.zeros_like(particle_types)
                    downsample_mask[particle_types == 1] = 1
                    downsample_mask[batch_idx, scene_downsample_idx] = 1
                    downsample_mask = downsample_mask.bool()

                    particles = particles[downsample_mask].reshape(B, -1, T, 3)
                    actions = actions[downsample_mask].reshape(B, -1, T-1, 3)
                    particle_types = particle_types[downsample_mask].reshape(B, -1)

                    # inject noise
                    if phase == 'train':
                        with torch.no_grad():
                            noise = torch.randn_like(particles) * dataset_config['randomness']['state_noise']['train']    # (B, N, T, 3)
                            noise[particle_types == 1, :, num_history:] = 0 # no noise on hand particles and future particles
                            particles = particles + noise
                        
                    loss = 0
                    for fi in range(dataset_config['n_future']):
                        this_particles = particles[:, :, fi:fi+num_history].clone()
                        assert this_particles.size(2) == num_history
                        this_actions = actions[:, :, num_history+fi-1].clone()
                        B, N, T, _ = this_particles.shape
                        this_model_input = torch.cat([
                            this_particles.view(B, N, T * 3),   # (B, N, 3T)
                            this_actions,                       # (B, N, 3)
                            particle_types.view(B, N, 1)        # (B, N, 1)
                        ], dim=-1)                              # (B, N, 3T+4)
                        this_pred_particles = model(this_model_input)   # (B, N, 3)
                        this_gt_particles = particles[:, :, num_history+fi]
                        this_loss = mse_loss(this_pred_particles, this_gt_particles)
                        loss += this_loss
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if i % train_config['log_interval'] == 0:
                            loss_sum_list.append(loss.item())
                            print(f"Epoch {epoch}, iter {i}/{n_iters}, loss: {loss.item()}")
                    
                    if phase == 'valid':
                        loss_sum_list.append(loss.item())

                if phase == 'train':
                    loss_plot_list_train.append(np.mean(loss_sum_list))
                if phase == 'valid':
                    loss_plot_list_valid.append(np.mean(loss_sum_list))
                    print(f"Epoch {epoch}, valid loss: {np.mean(loss_sum_list)}")
                    
        # save checkpoint
        torch.save(model.state_dict(), os.path.join(dyn_out_dir, "latest.pth"))    
        torch.save(optimizer.state_dict(), os.path.join(dyn_out_dir, "latest_optim.pth"))
        # save the best model
        if loss_plot_list_valid[-1] < best_valid_error:
            best_valid_error = loss_plot_list_valid[-1]
            torch.save(model.state_dict(), os.path.join(dyn_out_dir, "best.pth"))
            torch.save(optimizer.state_dict(), os.path.join(dyn_out_dir, "best_optim.pth"))
        
        # plot figures
        plt.figure(figsize=(20, 5))
        plt.plot(loss_plot_list_train, label='train')
        plt.plot(loss_plot_list_valid, label='valid')
        # cut off figure
        ax = plt.gca()
        y_min = min(min(loss_plot_list_train), min(loss_plot_list_valid))
        y_min = min(loss_plot_list_valid)
        y_max = min(3 * y_min, max(max(loss_plot_list_train), max(loss_plot_list_valid)))
        ax.set_ylim([0, y_max])
        # save
        plt.legend()
        plt.savefig(os.path.join(dyn_out_dir, 'loss.png'), dpi=300)
        plt.close()
        
        time2 = time.time()
        print(f'Epoch {epoch} time: {time2 - time1}\n')      


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/dexdeform_folding.yaml')
    args = arg_parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    train(config)