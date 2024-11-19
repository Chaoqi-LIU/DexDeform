import numpy as np
import os
import glob
import imageio
import open3d as o3d
from process_dataset import SAVE_DIR


def visualize_scene(
    scene_points: np.ndarray,
    hand_points: np.ndarray,
    hand_edges: np.ndarray,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.concatenate([hand_points, scene_points], axis=0))
    line_set = o3d.geometry.LineSet(
        points=pcd.points,
        lines=o3d.utility.Vector2iVector(hand_edges),
    )
    o3d.visualization.draw_geometries([pcd, line_set])


if __name__ == '__main__':
    demo_files = glob.glob(os.path.join(SAVE_DIR, "*.npy"))
    demo_files.sort()
    print(f"Found {len(demo_files)} demos")

    for demo_file in demo_files:
        print(f"Reading {demo_file}")

        demo_data = np.load(demo_file, allow_pickle=True).item()
        scene_points = demo_data["scene_points"]
        hand_points = demo_data["hand_points"]
        hand_edges = demo_data["hand_edges"]

        # load images
        images = demo_data["images"]
        front_images = images["front"]
        side_images = images["side"]
        top_images = images["top"]
        bot_images = images["bot"]
        concat_images = np.concatenate([
            np.concatenate([front_images, side_images], axis=2),
            np.concatenate([top_images, bot_images], axis=2)
        ], axis=1)
        imageio.mimsave("test.gif", concat_images, fps=20)
        # exit(1)

        for i in range(len(scene_points)):
            visualize_scene(
                scene_points[i],
                hand_points[i],
                hand_edges[i],
            )

