import numpy as np
import torch
import glob
import os
import gc
from pytorch3d.ops import sample_farthest_points

from policy.utils.io import load_gzip_file
from mpm import make
from mpm.viewer import Viewer


"""
process raw data into the form:
{
    "scene_points": np.ndarray,             # (T,   N,  3)
    "scene_points_velocity": np.ndarray,    # (T,   N,  3)
    "hand_points": np.ndarray,              # (T,   19, 3)
    "hand_edges": np.ndarray,               # (     72, 2)
    "action": np.ndarray,                   # (T-1, 24,  )
    "images": {
        "front": np.ndarray,                # (T,   H, W, 3)
        "side": np.ndarray,                 # (T,   H, W, 3)
        "top": np.ndarray,                  # (T,   H, W, 3)
        "bot": np.ndarray,                  # (T,   H, W, 3)
    }
}
where
    T: number of frames
    N: number of scene points, default 1024
    H: image height, set in the config yaml, default 512
    W: image width, set in the config yaml, default 512
"""

ENV_NAME = "folding"
NUM_SCENE_POINTS = 1024
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
DEMO_DIR = f"data/{ENV_NAME}"
SAVE_DIR = f"data/processed_{ENV_NAME}"


def dict_apply(
    x: dict,
    func: callable,
) -> dict:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


if __name__ == '__main__':
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    demo_files = glob.glob(os.path.join(DEMO_DIR, "*.pkl"))
    demo_files.sort()
    print(f"Found {len(demo_files)} demos")

    for demo_file in demo_files:
        print(f"Processing {demo_file}")

        processed_demo = {
            "scene_points": [],
            "scene_points_velocity": [],
            "hand_points": [],
            "hand_edges": [],
            "action": [],
            "images": {
                "front": [],
                "side": [],
                "top": [],
                "bot": [],
            }
        }

        demo_data = load_gzip_file(demo_file)
        for step_data in demo_data["states"]:
            scene_points = np.asarray(step_data[0])
            scene_points_velocity = np.asarray(step_data[1])
            hand_points = np.stack(step_data[4:-2])[:, :3]
            hand_edges = RIGHT_HAND_EDGES

            # sample scene points
            _, downsample_indices = sample_farthest_points(
                torch.from_numpy(scene_points).unsqueeze(0).cuda(),
                K=NUM_SCENE_POINTS, 
                random_start_point=True
            )
            downsample_indices = downsample_indices.cpu().numpy().squeeze(0)
            scene_points = scene_points[downsample_indices]
            scene_points_velocity = scene_points_velocity[downsample_indices]

            processed_demo["scene_points"].append(scene_points)
            processed_demo["scene_points_velocity"].append(scene_points_velocity)
            processed_demo["hand_points"].append(hand_points)
            processed_demo["hand_edges"].append(hand_edges)

        env = make(env_name=ENV_NAME, sim_cfg={'max_steps': 4400})
        viewer = Viewer(env)
        viewer.refresh_views("hand_centric")
        viewer.set_view("side")

        images = viewer.render_state_multiview(n_views=4, concat=False)
        processed_demo["images"]["front"].append(images[0])
        processed_demo["images"]["side"].append(images[1])
        processed_demo["images"]["top"].append(images[2])
        processed_demo["images"]["bot"].append(images[3])
        for action in demo_data["actions"]:
            processed_demo["action"].append(np.asarray(action[0]))  # single hand
            env.simulator.step(action)
            images = viewer.render_state_multiview(n_views=4, concat=False)
            processed_demo["images"]["front"].append(images[0])
            processed_demo["images"]["side"].append(images[1])
            processed_demo["images"]["top"].append(images[2])
            processed_demo["images"]["bot"].append(images[3])
        
        processed_demo = dict_apply(
            processed_demo,
            lambda x: np.stack(x),
        )

        print("================================")
        for k, v in processed_demo.items():
            if isinstance(v, np.ndarray):
                print(f"    {k}: {v.shape}")
            else:
                for kk, vv in v.items():
                    print(f"    {k}.{kk}: {vv.shape}")
        print("================================")

        # save
        save_file = os.path.join(SAVE_DIR, os.path.basename(demo_file))
        print(f"Saving to {save_file}")
        np.save(save_file, processed_demo)
        
        # cleanup (this is necessary)
        del env
        del viewer
        gc.collect()
