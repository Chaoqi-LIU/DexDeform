import os
import glob
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset


class GNNDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        phase='train'
    ):
        assert phase in ['train', 'valid', 'test']
        print(f'[{phase}] loading dataset...')
        self.phase = phase
        
        self.dataset_config = dataset_config
        
        self.n_his = dataset_config['n_his']
        self.n_future = dataset_config['n_future']
        
        self.add_randomness = dataset_config['randomness']['use']
        self.state_noise = dataset_config['randomness']['state_noise'][self.phase]
        
        self.pair_lists, self.obj_particles, self.hand_particles = self._load_data()
        print(f"[{self.phase}] dataset length: {self.pair_lists.shape[0]}. \n")
        
        
    
    def _load_data(self):
        """Load data
        Returns:
            pair_lists: numpy array (T, 8)
            obj_particles: list of object particles (n_epis, n_frames, N, 3)
            hand_particles: list of hand particles (n_epis, n_frames, 19, 3)
        """
        data_name = self.dataset_config['data_name']
        raw_data_dir = os.path.join(self.dataset_config['data_dir'], data_name, self.phase)
        prep_data_dir = os.path.join(self.dataset_config['prep_data_dir'], data_name, self.phase)
        
        self.raw_data_files = glob.glob(os.path.join(raw_data_dir, f'*.npy'))
        self.raw_data_files.sort()
        self.num_episodes = len(self.raw_data_files)
        print(f"[{self.phase}] Found {len(self.raw_data_files)} episodes.")
        
        # load pair list and states
        pair_lists = []        
        obj_particles = []
        hand_particles = []
        for episode_idx in range(self.num_episodes):
            # get object particles and hand particles
            demo_data = np.load(self.raw_data_files[episode_idx], allow_pickle=True).item()
            scene_points = demo_data['scene_points']  # (T, N, 3)
            hand_points = demo_data['hand_points']    # (T, 19, 3)
            obj_particles.append(scene_points)
            hand_particles.append(hand_points)
            
            # get pair list
            pair_data_path = os.path.join(prep_data_dir, 
                                          os.path.basename(self.raw_data_files[episode_idx]).replace('.npy', '.txt'))
            pair_list = np.loadtxt(pair_data_path).astype(int) # (T, n_his + n_future)
            
            # append episode idx to pair list
            pair_list = np.concatenate([np.full((pair_list.shape[0], 1), episode_idx), pair_list], axis=1)
            pair_lists.append(pair_list)
        
        # convert to numpy
        pair_lists = np.concatenate(pair_lists, axis=0)
        
        return pair_lists, obj_particles, hand_particles
    
    
    # def load_images(self):
    #     images_list = []
    #     for episode in range(self.num_episodes):
    #         demo_data = np.load(self.raw_data_files[episode], allow_pickle=True).item()
    #         images = demo_data['images']
            
    
    def __len__(self):
        return len(self.pair_lists)
    
    
    def __getitem__(self, index):
        episode_idx = self.pair_lists[index, 0].astype(int)
        pair = self.pair_lists[index, 1:].astype(int)
        T_seq = len(pair)
        assert T_seq == self.n_his + self.n_future
        
        # consturct particles
        obj_kps = []
        hand_kps = []
        for i in range(T_seq):
            frame_idx = pair[i]
            obj_kp = self.obj_particles[episode_idx][frame_idx]     # (1024, 3)
            hand_kp = self.hand_particles[episode_idx][frame_idx]   # (19, 3)
            obj_kps.append(obj_kp)
            hand_kps.append(hand_kp)
        obj_kps = np.stack(obj_kps, axis=0)     # (T_seq, 1024, 3)
        hand_kps = np.stack(hand_kps, axis=0)   # (T_seq, 19, 3)
        
        particles = np.concatenate([obj_kps, hand_kps], axis=1) # (T_seq, 1043, 3)
        num_obj_ptcls = obj_kps.shape[1]
        num_ptcls = particles.shape[1]
        
        # consturct attributes
        # obj: 0, hand: 1
        attributes = np.zeros((num_ptcls))
        attributes[num_obj_ptcls:] = 1
        
        # construct actions
        actions = np.zeros((T_seq-1, particles.shape[1], 3))
        for fi in range(T_seq-1):
            actions[fi, num_obj_ptcls:] = particles[fi+1, num_obj_ptcls:] - particles[fi, num_obj_ptcls:]
        
        # consturct num_particles_per_sample
        num_particles_per_sample = np.array([num_ptcls])
        
        # reshape to (N, T, 3) for particles and actions
        particles = particles.transpose(1, 0, 2)
        actions = actions.transpose(1, 0, 2)
        
        # numpy to torch
        particles = torch.from_numpy(particles).float()
        attributes = torch.from_numpy(attributes).float()
        actions = torch.from_numpy(actions).float()
        num_particles_per_sample = torch.from_numpy(num_particles_per_sample).long()
        
        data = {
            'particles': particles,                                 # (B, N, T, 3)
            'particle_types': attributes,                           # (B, N)    
            'actions': actions,                                     # (B, N, T-1, 3)    
            'num_particles_per_sample': num_particles_per_sample    # (B,)
        }
        
        return data
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dexdeform_folding.yaml')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    dataset_config = config['dataset_config']
    
    dataset = GNNDataset(dataset_config, phase='train')
    data = dataset[0]
    
