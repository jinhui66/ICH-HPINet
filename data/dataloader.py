from __future__ import division
import json
from logging import warning
import os
from networkx.algorithms.assortativity import connectivity
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from davisinteractive.utils.scribbles import scribbles2mask,annotated_frames
from davisinteractive.robot.interactive_robot import InteractiveScribblesRobot
from skimage.morphology import disk, binary_erosion, binary_dilation, binary_opening, ball
from skimage.filters import threshold_otsu, gaussian
from skimage import measure
from utils import generate_guid_map
import pickle
import torch.nn.functional as F
import warnings
import SimpleITK as sitk
import torch 
####################################################################################################################
####################################################################################################################


class Train_Prop(Dataset):
    """
    training dataloader for propagation model
    """
    def __init__(self,phase='train',
                preprocessed_data_path="/mnts2d/med_data1/lyshi/KiTS_nnUNet_preprocessed/",# preoricessed data path
                patch_data_path = "/mntnfs/med_data2/shiluyue/InteractiveCTSeg/dataset/KiTS_Organ_3D_Patch",
                transform={"flip":0.5, "crop":0.5},
                train_sample_list = "./data/public_train.txt", # list contrains case IDs or string (path to txt file), case IDs for training
                object = "organ", # 1 or 2, choose organ or tumer as the target object
                data_info_file = "/data/lyshi/datasets/nnUNet_data/nnUnet_preprocessed/Task050_KiTS/nnUNetData_plans_v2.1_2D_stage0/data_info.json",
                patch_out_size = 80
                ):
        self.phase=phase
        self.data_root_dir=preprocessed_data_path
        self.data_patch_dir = patch_data_path
        self.transform=transform
        self.patch_out_size = patch_out_size
        self.process = 0 # 0~1, current_epoch / total_epoch
        # according to the MiVOS, the maximum distance between frames increased from 5 to 25 gradually (curriculum learning) and annealed back to 5 toward the end of training
        # in this project, the maximum distance is 0.1length-0.4length-0.1length

        # if self.data_patch_dir is not None:
        #     self.crop_info = pickle.load(os.path.join(self.data_patch_dir, "meta.pkl"))

        self.target_object = object
        if self.target_object not in ["organ", "tumor"]:
            raise ValueError("The target_object-{} does not exist, which should be 'organ' or 'tumor'.".format(object))

        if type(train_sample_list) == type([]):
            self.train_id_list = train_sample_list
        elif type(train_sample_list) == type(''):
            with open(train_sample_list, "r") as f:
                id_list = f.readlines()
            self.train_id_list = []
            for idx in id_list:
                self.train_id_list.append(idx[:-1])
        else:
            print("train_sample_list should be a list or path")
        

        self.sample_list=[]
        self.mask_info = {}
        self.total_file_num = len(self.train_id_list)
        count = 0
        print("Looking through the whole dataset...")
    
        self.current_case = None

        self.interactor = InteractiveScribblesRobot(
                kernel_size=0.12,
                max_kernel_radius=16,
                min_nb_nodes=4,
                nb_points=1000
            )

    def __len__(self):
        # return len(self.sample_list)
        return self.total_file_num
    
    def __getitem__(self,idx):
        case_id = self.train_id_list[idx]
        if case_id != self.current_case:
            self.current_case = case_id
            volume = sitk.GetArrayFromImage(sitk.ReadImage(f'{self.data_root_dir}/preprocessed_image/{case_id}.nii.gz'))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(f'{self.data_root_dir}/binary_mask/{case_id}.nii.gz'))

            # print(volume.shape[0])
            volume = volume[len(volume)//2 - 8 : len(volume)//2 + 8]
            mask = mask[len(mask)//2 - 8 : len(mask)//2 + 8]
            # print(np.max(mask))

            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            # print(max(mask), min(mask))
            assert(np.sum(mask==1)>0)

            # if np.random.rand() < self.transform["flip"]:
            #     flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
            #     volume = np.ascontiguousarray(volume[::flipid[0],::flipid[1],::flipid[2]])
            #     mask = np.ascontiguousarray(mask[::flipid[0],::flipid[1],::flipid[2]])
            # if np.random.rand() < self.transform["crop"]:
            #     shape = volume.shape
            #     start = np.random.randint(0, [int(i*0.15) for i in shape], 3)
            #     end = np.random.randint(1, [int(i*0.15) for i in shape], 3)
            #     lh = shape[1] - end[1] - start[1]
            #     start[2] = np.random.randint(0, max(min(shape[2] - lh, int(shape[1]*0.15)), 1))
            #     end[2] = max(shape[2] - start[2] - lh, 1)
            #     volume = volume[:, start[1]:-end[1], start[2]:-end[2]]
            #     mask = mask[:, start[1]:-end[1], start[2]:-end[2]]
            
            assert(np.sum(mask==1)>0)

            d = volume.shape[-1] % 16
            if d!=0:
                try:
                    volume = volume[:,:,:-d]
                    mask = mask[:,:,:-d]
                    assert(np.sum(mask==1)>0)
                except:
                    volume = np.pad(volume, ((0,0),(0,0),(0,16-d)), 'constant')
                    mask = np.pad(mask, ((0,0),(0,0),(0,16-d)), 'constant')
            d = volume.shape[-2] % 16
            if d!=0:
                try:
                    volume = volume[:,:-d,:]
                    mask = mask[:,:-d,:]
                    assert(np.sum(mask==1)>0)
                except:
                    volume = np.pad(volume, ((0,0),(0,16-d),(0,0)), 'constant')
                    mask = np.pad(mask, ((0,0),(0,16-d),(0,0)), 'constant')

            assert(np.sum(mask==1)>0)

            volume = torch.from_numpy(volume)
            mask = torch.from_numpy(mask)
            
            self.volume = volume
            self.mask = mask

            # print(torch.max(volume), torch.max(mask))
            self.volume_patch, self.mask_patch, self.crop_info, self.prev_round_mask, self.scribble_mask = self.crop_3d_patch(volume, mask)


            # print(self.scribble_mask.shape)
            z_max = torch.max(torch.where(self.mask==1)[0])
            # assert(z_max <= self.crop_info["crop_range"][3]) 


        slice_index = torch.unique(torch.where(self.mask==1)[0])
        slice_num = len(slice_index)
        if self.process < 0.5:
            max_d = 3/5 * self.process + 0.1
        else:
            max_d = -3/5 * self.process + 7/10
        max_d = int(np.round(slice_num * max_d))
        max_d = max(max_d,2)

        direct = np.random.randn()
        if slice_num < 3:
            print("Error! {} has {} slices, cannot generate training sample \
                which requires three slices.".format(case_id, slice_num))
            return
        elif slice_num == 3:
            start_idx, slice_1_idx, slice_2_idx = [0, 1, 2] if direct>0 else [2, 1, 0]
        else:
            start_idx = np.random.randint(0, slice_num)
            if start_idx < 2 or (direct < 0 and start_idx < slice_num-2):
                slice_1_idx = np.random.randint(start_idx+1, min(slice_num-1, start_idx+max_d))
                slice_2_idx = np.random.randint(slice_1_idx+1, min(slice_num, start_idx+max_d+1))
            else:
                slice_1_idx = np.random.randint(max(1, start_idx-max_d+1), start_idx)
                slice_2_idx = np.random.randint(max(0, start_idx-max_d), slice_1_idx)
        # print(self.volume.shape, self.scribble_mask.shape, self.prev_round_mask.shape)
        start_slice = self.volume[slice_index[start_idx]].clone().unsqueeze(0)
        start_slice_mask = self.mask[slice_index[start_idx]].clone().unsqueeze(0)
        slice_scribble = self.scribble_mask[:,slice_index[start_idx]].clone().unsqueeze(0)
        slice_scribble = F.interpolate(slice_scribble, size=[512,512], mode="bilinear", align_corners=False)[0]
        slice_scribble[slice_scribble >= 0.5] = 1
        slice_scribble[slice_scribble < 0.5] = 0
        # print(torch.max(self.scribble_mask[1,slice_index[start_idx]]), torch.min(self.scribble_mask[1,slice_index[start_idx]]))
        slice_prev_mask = self.prev_round_mask[0, slice_index[start_idx]].clone().unsqueeze(0).unsqueeze(0)
        slice_prev_mask = F.interpolate(slice_prev_mask, size=[512,512], mode="bilinear", align_corners=False)[0]
        slice_prev_mask[slice_prev_mask >= 0.5] = 1
        slice_prev_mask[slice_prev_mask < 0.5] = 0

        slice_1 = self.volume[slice_index[slice_1_idx]].clone().unsqueeze(0)
        slice_1_mask = self.mask[slice_index[slice_1_idx]].clone().unsqueeze(0)
        slice_2 = self.volume[slice_index[slice_2_idx]].clone().unsqueeze(0)
        slice_2_mask = self.mask[slice_index[slice_2_idx]].clone().unsqueeze(0)
        # print(slice_scribble.shape, slice_prev_mask.shape)

        for img in [start_slice,start_slice_mask,slice_1,slice_1_mask,slice_2,slice_2_mask]:
            if len(img.shape) != 3:
                print("Incorrect image shape, got ", img.shape)

        sample = {
            "start_slice": start_slice,
            "start_slice_mask": start_slice_mask,
            "slice_scribble": slice_scribble,
            "slice_prev_mask": slice_prev_mask,
            "slice_1": slice_1,
            "slice_1_mask": slice_1_mask,
            "slice_2": slice_2,
            "slice_2_mask": slice_2_mask,
            "meta":{"caseID": case_id,"slice_idx":[slice_index[start_idx], slice_index[slice_1_idx],slice_index[slice_2_idx]]},
            "volume_patch": self.volume_patch,
            "mask_patch": self.mask_patch,
            "crop_info": self.crop_info,
            "prev_round_mask": self.prev_round_mask, 
            "scribble_mask": self.scribble_mask,
        }
        return sample

    def crop_3d_patch(self, volume, mask):
        if type(volume)==np.ndarray:
            volume = torch.from_numpy(volume)
        if type(mask)==np.ndarray:
            mask = torch.from_numpy(mask)
        # z, h, w = volume.shape
        # region = torch.where(mask==1)
        # corners = torch.FloatTensor([torch.min(region[i]) for i in range(3)] + [torch.max(region[i]) for i in range(3)])
        # center = torch.FloatTensor([torch.mean(corners[[i, i+3]]) for i in range(3)])
        # lz, lh, lw = [corners[i+3] - corners[i] for i in range(3)]
        # crop_length_hw, crop_length_z = [self.patch_out_size, self.patch_out_size]

        # crop_length_hw = max(max(lh,lw)+10, self.patch_out_size)
        # # if crop_length_hw%16!=0: 
        # #     crop_length_hw = crop_length_hw + 16 - crop_length_hw%16
        # crop_length_z = lz + 10

        # # if lz+16-lz%16 <= z:
        # #     crop_length_z = lz+16-lz%16
        # #     z_padding = 0
        # # else:
        # #     crop_length_z = z
        # #     z_padding = lz+16-lz%16-z
        # # crop_length_z = int(crop_length_z)
        # c1 = corners.clone()

        # corners[1:3] = center[1:] - crop_length_hw / 2
        # corners[4:6] = corners[1:3] + crop_length_hw
        # corners[1:3] = corners[1:3] - corners[1:3]%16
        # corners[4:6] = corners[4:6] + torch.where(16-corners[4:6]%16<16, 16-corners[4:6]%16, torch.zeros(2))
        # corners[0] = center[0] - crop_length_z / 2
        # corners[3] = corners[0] + crop_length_z
        # corners[0] = corners[0] - corners[0]%16
        # corners[3] = corners[3] + 16 - corners[3]%16 if corners[3]%16>0 else corners[3]
        # c2 = corners.clone()

        # for i in range(3):
        #     l = [z,h-h%16,w-w%16][i]
        #     if corners[i+3] > l and i!=0:
        #         corners[i] -= corners[i+3]-l
        #         corners[i+3] = l
        #     if corners[i] < 0:
        #         corners[i+3] -= corners[i]
        #         corners[i] = 0
        #     if corners[i+3] > l and i!=0:
        #         corners[i+3] = l
        
        # z_padding = 0
        # if corners[3] > z:
        #     z_padding = corners[3] - z
        #     corners[3] = z

        # c3 = corners.clone()
        # assert(torch.all(corners[:3]<=c1[:3]))
        # assert(torch.all(corners[3:]>=c1[3:]))

        # corners = corners.long()

        # assert(torch.all(corners>=0))
        # assert(corners[3]<=z)
        # assert(corners[4]<=h)
        # assert(corners[5]<=h)
        # if not torch.all(corners[[0,1,2,4,5]]%16==0):
        #     print("====", c1, c2, c3, volume.shape)
        # assert(torch.all(corners[[0,1,2,4,5]]%16==0))
        
        # size = [int(corners[3]-corners[0]+z_padding), int(corners[4]-corners[1]), int(corners[5]-corners[2])]
        # volume_patch = torch.zeros(size)
        # mask_patch = torch.zeros(size)
        # volume_patch[:corners[3]-corners[0]] = volume[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]]
        # mask_patch[:corners[3]-corners[0]] = mask[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]]
        
        volume_patch = volume.float()
        mask_patch = mask.float()
        
        mask_ori = mask_patch.clone()

        crop_info = {"ori_crop_size":torch.tensor(volume_patch.shape), "volume_size":torch.tensor(volume.shape), "crop_range":torch.tensor([0,0,0,volume.shape[0],volume.shape[1],volume.shape[2]])}

        volume_patch = volume_patch.unsqueeze(0).unsqueeze(0)
        volume_patch = F.interpolate(volume_patch, size=[16,256,256], mode="trilinear", align_corners=False)[0]
        mask_patch = mask_patch.unsqueeze(0).unsqueeze(0)
        mask_patch = F.interpolate(mask_patch, size=[16,256,256], mode="trilinear", align_corners=False)[0]


        # print(mask_ori.shape, mask_patch.shape)
        prev_round_mask, scribble_mask = self.generate_scribble(mask_ori, mask_patch[0])
        # _, full_scribble_mask = self.generate_scribble(mask, mask)

        return volume_patch, mask_patch, crop_info, prev_round_mask, scribble_mask
    def generate_scribble(self, mask_ori, mask_resize):
        z, h, w = mask_ori.shape
        mask_resize = torch.where(mask_resize>0.5, torch.ones_like(mask_resize), torch.zeros_like(mask_resize)).numpy()
        prev_round_mask_resize, generation_mode = self.generate_prev_round_mask(mask_resize)
        prev_round_mask_resize = torch.from_numpy(prev_round_mask_resize.astype(np.float32)).unsqueeze(0)
        prev_round_mask_ori = F.interpolate(prev_round_mask_resize.unsqueeze(0), size=[z,h,w], mode="trilinear", align_corners=False)[0,0]

        # generate scribble mask
        if generation_mode == "zero" or len(np.where(prev_round_mask_ori==1)[0])==0:
            z_range = np.unique(np.where(mask_ori==1)[0])
            int_z = np.random.choice(range(int(len(z_range)*0.2), int(len(z_range)*(1-0.2))))
            int_z = z_range[int_z]
        else:
            z_range = np.unique(np.where(prev_round_mask_ori==1)[0])
            int_z = np.random.choice(z_range)
        int_pred = prev_round_mask_ori.numpy().copy()
        int_pred[int_pred>0.5] = 1
        int_pred[int_pred!=1] = 0
        int_gt = mask_ori.numpy().copy()
        int_gt[int_gt>0.5] = 1
        int_gt[int_gt!=1] = 0
        assert(len(np.unique(np.where(int_gt==1)[0]))>0)
        scribble_dict = self.interactor.interact(sequence='', pred_masks=int_pred, gt_masks=int_gt, nb_objects=1, frame=int_z)
        scrib_mask = scribbles2mask(scribble_dict, output_resolution=(h,w))
        if len(np.unique(scrib_mask))==0:
            warnings.warn("scribble_mask is zero.")
        scribble_mask = np.zeros((2,z,h,w))
        scribble_mask[0][np.where(scrib_mask==0)] = 1 # negative
        scribble_mask[1][np.where(scrib_mask==1)] = 1
        scribble_mask = torch.from_numpy(scribble_mask.astype(np.float32))
        scribble_mask_resize = F.interpolate(scribble_mask.unsqueeze(0), size=mask_resize.shape, mode="trilinear", align_corners=False)[0]

        return prev_round_mask_resize, scribble_mask_resize

    def generate_prev_round_mask(self, gt_mask, zero_map_prob=0.5, alpha_m2=0.5, alpha_md=0.1, alpha_m1=0.3):
        z, h, w = gt_mask.shape
        generation_mode = np.random.choice(["morphology","morphology2","neighbor"])
        if np.random.rand() < zero_map_prob:
            mask = np.zeros((z,h,w))
            generation_mode = "zero"
        elif generation_mode in ["morphology", "morphology2"]:
            thresh = threshold_otsu(gt_mask)
            gt_mask[gt_mask>thresh] = 1
            gt_mask[gt_mask!=1] = 0
            max_i = (-1, 0)
            for i in range(z):
                area = np.sum(gt_mask[i])
                if area > max_i[1]: max_i = (i, area)
            mask_range = np.where(gt_mask[max_i[0]]==1)
            r = np.min([np.max(mask_range[i]) - np.min(mask_range[i]) for i in range(2)]) / 2
            if generation_mode == "morphology":
                max_ball_r = int(np.round(r * alpha_m1))
                if max_ball_r <= 1:
                    ball_r = 1
                else:
                    ball_r = np.random.randint(1, max_ball_r+1)
                if np.random.rand() > 0.5:
                    mask = binary_erosion(gt_mask, ball(ball_r))
                else:
                    mask = binary_dilation(gt_mask, ball(ball_r))
            elif generation_mode == "morphology2":
                ball_r = np.random.randint(1, max(int(np.round(r * alpha_m2)), 2))
                mask = binary_opening(gt_mask, ball(ball_r))

            mask = np.array(mask, dtype=np.float32)
        elif generation_mode=="neighbor":
            z_range = np.where(gt_mask==1)[0]
            z_len = np.max(z_range) - np.min(z_range)
            move_d = np.random.choice(range(1, max(int(z_len * alpha_md), 2)))
            mask = np.zeros(gt_mask.shape)
            if np.random.randn() > 0:
                mask[move_d:] = gt_mask[:-move_d]
                mask[:move_d] = np.repeat(gt_mask[0:1], move_d, axis=0)
            else:
                mask[:-move_d] = gt_mask[move_d:]
                mask[-move_d:] = np.repeat(gt_mask[-1:], move_d, axis=0)

        return mask, generation_mode


####################################################################################################################
####################################################################################################################
