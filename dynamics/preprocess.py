import os
import glob
import numpy as np
import argparse
import time
import yaml


def preprocess(config):
    ## config
    dataset_config = config["dataset_config"]
    task_name = dataset_config["data_name"]
    
    raw_data_dir = os.path.join(dataset_config["data_dir"], task_name)
    prep_data_dir = os.path.join(dataset_config["prep_data_dir"], task_name)
    os.makedirs(prep_data_dir, exist_ok=True)
    
    n_his = dataset_config["n_his"]
    n_future = dataset_config["n_future"]
    
    phases = ["train", "valid"]
    for phase in phases:
        print(f"[{phase}] preprocessing...")
        
        save_dir = os.path.join(prep_data_dir, phase)
        os.makedirs(save_dir, exist_ok=True)
        
        data_files = glob.glob(os.path.join(raw_data_dir, phase, f"*.npy"))
        data_files.sort()
        print(f"[{phase}] Found {len(data_files)} episodes.")
        
        # loop episodes
        for data_file in data_files:
            demo_data = np.load(data_file, allow_pickle=True).item()
            scene_points = demo_data["scene_points"] # (T, N, 3)
            num_frames = scene_points.shape[0]
            print(f"[{phase}] {data_file}: {num_frames} frames")
            
            frame_idxs = []
            start_frame = 0
            end_frame = num_frames
            
            # loop frames
            for fj in range(num_frames):
                # init
                curr_frame = fj
                frame_traj = [curr_frame]
                
                # search backward (n_his)
                fi = fj - 1
                while fi >= start_frame:
                    frame_traj.append(fi)
                    fi -= 1
                    if len(frame_traj) == n_his:
                        break
                else:
                    # pad to n_his
                    frame_traj = frame_traj + [frame_traj[-1]] * (n_his - len(frame_traj))
                frame_traj = frame_traj[::-1]
                
                # search forward (n_future)
                fi = fj + 1
                while fi <  end_frame:
                    frame_traj.append(fi)
                    fi += 1
                    if len(frame_traj) == n_his + n_future:
                        break
                else:
                    # pad to n_future
                    frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - len(frame_traj))
                
                # append to frame_idxs list
                frame_idxs.append(frame_traj)
            
            # to numpy and save
            frame_idxs = np.array(frame_idxs)
            np.savetxt(os.path.join(save_dir, os.path.basename(data_file).replace(".npy", ".txt")), frame_idxs, fmt='%d')
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dexdeform_folding.yaml')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    preprocess(config)
    