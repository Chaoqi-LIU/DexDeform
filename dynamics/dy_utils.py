import numpy as np
import random
import torch

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        # print(f'[{name}] epoch {cnt}')
        cnt += 1
        for data in dataloader:
            yield data

def grad_manager(phase):
    if phase == 'train':
        return torch.enable_grad()
    else:
        return torch.no_grad()
    
    