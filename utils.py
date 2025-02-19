import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, sample_list, num_instances=1):
        self.sample_list = sample_list
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, tmp_dic in enumerate(self.sample_list):
            pid = tmp_dic[0][-14:-4]
            # pid = tmp_dic['seq_name']
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

class PropagationRandomSampler(Sampler):
    """
    Randomly sample N intances for each indentity.
    The default N equals to batch size, which means each batch contrains N samples which are from the same case (same CT volume).
    """

    def __init__(self, sample_list, num_instances=1):
        self.sample_list = sample_list
        self.num_instances = num_instances

        self.num_identities = len(self.sample_list)

    def __iter__(self):
        idx_list = np.arange(self.num_identities)
        np.random.shuffle(idx_list)
        idx_list = np.repeat(idx_list, self.num_instances)
        idx_list = list(idx_list)
        return iter(idx_list)

    def __len__(self):
        return self.num_identities * self.num_instances

def pad_divide_by(in_img, d, in_size=None):
    if in_size is None:
        h, w = in_img.shape[-2:]
    else:
        h, w = in_size

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array


class TestIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, sample_list, batch_size=1):
        self.sample_list = sample_list
        self.batch_size = batch_size
        self.index_dic = defaultdict(list)
        for index, tmp_dic in enumerate(self.sample_list):
            pid = tmp_dic[0][-14:-4]
            # pid = tmp_dic['seq_name']
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = range(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            # print(pid, len(t))
            if len(t) % self.batch_size == 0:
                ret.extend(t)
            else:
                add = self.batch_size - len(t)%self.batch_size
                add_t = np.random.choice(t, size=add, replace=False)
                ret.extend(t)
                ret.extend(add_t)
        self.ret = ret
        return iter(ret)

    def __len__(self):
        return len(self.ret)


import cv2
import numpy as np

# def generate_guid_map(mask, center_idx):
#     # mask: segmentation
#     d, h, w = mask.shape
#     mask[mask>0.5] = 1
#     mask[mask<=0.5] = 0

#     guid_map = np.zeros(h,w)
#     for i in range(11):
#         slice = mask[center_idx+i]
#         contours, hierarchy = cv2.findContours(slice,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
#     return

def generate_guid_map(volume, center_idx):
    return motion_history_img(volume, center_idx, param1=1, param2=0.1)

def motion_history_img(volume, center_idx, param1=1, param2=0.1):
    # mask: segmentation
    d, h, w = volume.shape
    # print(volume.shape, center_idx)

    mhi = np.zeros((h,w))
    prev_slice = volume[min(center_idx+10,d-1)]
    for i in range(10-1): # -1: exclude central slice
        if center_idx+9-i >= d:
            continue
        slice = volume[center_idx+9-i]
        mhi = update_mhi(slice, prev_slice, mhi, param1, param2)
        prev_slice = slice

    mhi_2 = np.zeros((h,w))
    prev_slice = volume[max(center_idx-10,0)]
    for i in range(10-1):
        if center_idx-9+i < 0:
            continue
        slice = volume[center_idx-9+i]
        mhi_2 = update_mhi(slice, prev_slice, mhi_2, param1, param2)
        prev_slice = slice
    
    return mhi, mhi_2

def update_mhi(cur_img, prev_img, prev_map, param1, param2):
    diff = cur_img - prev_img
    motion_region = diff >= param1
    prev_map[motion_region] = 1
    prev_map[~motion_region] -= param2
    prev_map[prev_map<0] = 0
    return prev_map
