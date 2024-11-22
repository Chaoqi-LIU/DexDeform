import numpy as np
import os
import glob
from PIL import Image
from process_dataset import SAVE_DIR


def gen_rollout_filmstrip(demo, nframes=6, padding=5):
    images = demo["images"]
    front_images = images["front"]
    side_images = images["side"]
    top_images = images["top"]
    bot_images = images["bot"]

    H, W = front_images[0].shape[:2]
    T = len(front_images)
    if nframes > T:
        nframes = T
    stride = T // nframes
    nrow = 4
    ncol = nframes
    filmstrip = Image.new(
        "RGB",
        (W * ncol + padding * (ncol - 1), H * nrow + padding * (nrow - 1)),
        color="white",
    )
    for i in range(nframes):
        front = Image.fromarray(front_images[i * stride])
        side = Image.fromarray(side_images[i * stride])
        top = Image.fromarray(top_images[i * stride])
        bot = Image.fromarray(bot_images[i * stride])
        filmstrip.paste(front, (i * W + i * padding, 0))
        filmstrip.paste(side, (i * W + i * padding, H + padding))
        filmstrip.paste(top, (i * W + i * padding, 2 * H + 2 * padding))
        filmstrip.paste(bot, (i * W + i * padding, 3 * H + 3 * padding))
    return filmstrip


if __name__ == '__main__':
    demo_files = glob.glob(os.path.join(SAVE_DIR, "*vis.npy"))
    demo_files.sort()

    filmstrip = gen_rollout_filmstrip(np.load(demo_files[1], allow_pickle=True).item())
    filmstrip.save("sim_rollout.pdf", format="PDF")
    