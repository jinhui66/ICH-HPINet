from multiprocessing.sharedctypes import Value
from operator import add
from os import wait
import numpy as np
import torch
import torch.nn as nn
import cv2 
import torch.nn.functional as F
import warnings

from networks.deepLabV3.modeling import deeplabv3plus_resnet50
from networks.STCN.prop_net import PropagationNetwork, Converter
from utils import pad_divide_by

class iMIS:
    def __init__(self, seg3d_model, int_model, prop_model, volumn=None, num_objects=1, gpu=True, f3d_converter=Converter()):
        '''
        interactive medical image segmentation framework

        feature_extraction_model - 3d feature extraction model
        int_model - interaction model
        prop_model - propagation model
        volumn - whole volumn with shape D*H*W
        '''
        self.k = self.num_objects = num_objects
        self.volumn = volumn #cpu

        self.gpu = gpu
        self.device = "cuda" if gpu else "cpu"

        # parameters for cropping 3d patch
        self.crop_size = 96

        self.seg3d_model = seg3d_model
        self.int_model = int_model
        self.prop_model = prop_model
        self.f3d_converter = f3d_converter.to(self.device)

        self.segmented_index_list = []

        self.memory_keys = []
        self.memory_values = []

        self.current_round = 0
    
    def interact(self, scribble_mask, only_int=False, final_pred="prop", only_3d=False):
        '''
        interaction -> propagation -> 3d interaction in the 1st round, interaction -> 3d interaction -> propagation in other rounds.

        scribble_mask: scribble mask with shape D*H*W
        only_int: only run interaction model for current slice, no propagation
        final_pred: type of output mask, ["prop", "seg3d", "avg"](propagation output, segmentation output, average result)
        '''
        self.this_round_segmented_idx = []
        self.current_round += 1
        crop_3d = True

        # get the annotated slice, and scribble
        if np.all(scribble_mask==-1):
            print("No interaction information found, passed!")
            return self.get_pred_mask(final_pred), None
            
        scribble_mask, idx, scribble_3d = self.process_eval_scribble(scribble_mask) # 2*h*w, 1*2*d*h*w
        int_slice = self.volumn[idx]
        int_slice, scribble_mask, prev_r_mask, scribble_3d = self.prepare_tensor([int_slice, scribble_mask, self.pred_masks[idx], scribble_3d])
        assert(scribble_mask.shape[-1]==int_slice.shape[-1])

        self.only_3d = only_3d
        if only_3d:
            _ = self.run_3d_model(scribble_3d, crop_first=crop_3d, only_3d=only_3d)
            return self.get_pred_mask(final_pred), idx

        # In the first round, run 3d_model after propagation; in other rounds, run 3d_model before propagation.
        if self.current_round!=1:
            _ = self.run_3d_model(scribble_3d, crop_first=crop_3d)
        
        # interaction
        print("Interacting slice ", idx)
        # print(int_slice.shape)
        # print(scribble_mask.shape)
        # print(prev_r_mask.shape)
        self.pred_masks[idx] = self.run_int_model(int_slice, scribble_mask, prev_r_mask).cpu()
        # print(np.unique(self.pred_masks[idx].numpy()))
        
        if idx not in self.segmented_index:
            self.segmented_index.append(idx)
        if idx not in self.this_round_segmented_idx:
            self.this_round_segmented_idx.append(idx)
        # if idx not in self.interactive_slice:
        #     self.interactive_slice.append(idx)
        self.interactive_slice.append(idx)
        
        if only_int:
            return self.get_pred_mask(final_pred), idx
        
        # calculate the memory key and value using the interactive frame, which will be used during the propagation
        int_slice = self.volumn[idx]
        int_slice_mask = self.pred_masks[idx]
        int_slice, int_slice_mask = self.prepare_tensor([int_slice, int_slice_mask])
        m_k, _, m_f16, _, _ = self.prop_model.encode_key(int_slice) # m_k: b*64*h/16*w/16
        m_v = self.prop_model.encode_value(int_slice, m_f16, int_slice_mask) # m_v: b*512*1*h/16*w/16
        self.int_slice_keys.append(m_k.unsqueeze(2))
        self.int_slice_values.append(m_v)

        # propagation
        if idx==0:
            self.propagate(1, self.max_idx)
        elif idx==self.max_idx:
            self.propagate(self.max_idx-1, 0)
        else:
            self.propagate(idx+1, self.max_idx)
            self.propagate(idx-1, 0)

        if self.current_round==1:
            _ = self.run_3d_model(scribble_3d, crop_first=crop_3d)
        
        # print("Round finished!")
        return self.get_pred_mask(final_pred), idx
       
    def propagate(self, start, end):

        memory_keys = torch.cat(self.int_slice_keys, dim=2)
        memory_values = torch.cat(self.int_slice_values, dim=2)

        # propagate to other frames
        if start <= end:
            forward = True
            the_range = range(start, end+1)
        elif start > end:
            forward = False
            the_range = range(start, end-1, -1)
        first_frame = True
        zero_slice_count = 0
        print("Propagating slice ", end='')
        for idx in the_range:
            # stop when meet the interactive frames
            if idx in self.interactive_slice:
                break
            
            print("{}->".format(idx), end='')
            
            f_3d, _ = self.decode_3d_feature(idx, pred=False)
            
            query_slice = self.volumn[idx]
            query_slice = self.prepare_tensor([query_slice])[0]
            if first_frame:
                pred_mask, _ = self.run_prop_model(query_slice, f_3d, None, None, clear=first_frame, additional_k_v=[memory_keys, memory_values])
                first_frame = False
            else:
                prev_slice = self.volumn[idx-1] if forward else self.volumn[idx+1]
                prev_slice_mask = self.pred_masks[idx-1] if forward else self.pred_masks[idx+1]
                prev_slice, prev_slice_mask = self.prepare_tensor([prev_slice, prev_slice_mask])
                pred_mask, _ = self.run_prop_model(query_slice, f_3d, prev_slice, prev_slice_mask, clear=first_frame, additional_k_v=[memory_keys, memory_values])
            self.pred_masks[idx] = pred_mask.cpu()

            if np.sum(self.pred_masks[idx].numpy()>0.5)==0:
                zero_slice_count += 1
            if zero_slice_count > 0: #TODO
                print(" No object found, stop propagation. ", end='')
                break

            if idx not in self.segmented_index_list:
                self.segmented_index_list.append(idx)
            if idx not in self.this_round_segmented_idx:
                self.this_round_segmented_idx.append(idx)
        print("Down!")

    def run_3d_model(self, scribble_3d, crop_first=True, only_3d=False):
        '''
        scribble: b*2*D*H*W, same size as self.volume \\
        Crop and resize first, then feed to 3d-cnn. \\
        The output feature is then resized to the original cropped size before resize.
        '''
        # print("Executing 3d segmentation model...")
        # if self.current_round == 1:
        #     self.first_scribble = scribble_3d
        # else:
        #     scribble_3d = self.first_scribble
        if crop_first:
            scribble_3d = self.crop_3d_patch(scribble_3d)
            # scribble_3d[scribble_3d>0] = 1

        if self.volume_patch is None:
            raise ValueError("self.volume_patch not found, run crop_3d_patch first!")

        if only_3d:
            out, self.features_3d = self.seg3d_model(torch.cat([self.volume_patch, self.pred_patch_mask, scribble_3d],dim=1))
        else:
            if self.current_round<=1:
                # print(self.volume_patch.shape, self.pred_patch_mask.shape, scribble_3d.shape)
                out, self.features_3d = self.seg3d_model(torch.cat([self.volume_patch, self.pred_patch_mask, scribble_3d],dim=1))
            else:
                # corners = self.crop_info["crop_range"]
                # size = [1,1,int(corners[3]-corners[0]+self.crop_info["z_padding"]), int(corners[4]-corners[1]), int(corners[5]-corners[2])]
                patch_mask = self.pred_masks.unsqueeze(0).unsqueeze(0).to(self.device).float()
                # patch_mask[0,0,:corners[3]-corners[0]] = self.pred_masks[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]]
                patch_mask = F.interpolate(patch_mask, size=[16, 256, 256], mode="trilinear", align_corners=False)
                out, self.features_3d = self.seg3d_model(torch.cat([self.volume_patch, patch_mask, scribble_3d],dim=1))


        self.pred_patch_mask = torch.sigmoid(out)

        cropped_size = torch.tensor(self.crop_info["ori_crop_size"])
        self.features_3d_resize = {
            k: F.interpolate(self.features_3d[k], size=tuple(cropped_size//int(k[1:])), mode="trilinear", align_corners=False) for k in self.features_3d.keys()
        }

        return self.pred_patch_mask

    def decode_3d_feature(self, slice_idx, pred=False): # convert 3d feature to 2d slice feature
        if self.current_round == 1 and self.features_3d_resize is None:
            D, H, W = self.volumn.shape
            f_3d = {
                "f4": torch.zeros(1, 256, H//4, W//4).to(self.device),
                "f2": torch.zeros(1, 128, H//2, W//2).to(self.device),
                "f1": torch.zeros(1, 64, H, W).to(self.device),
            } 
            return f_3d, None

        if self.features_3d_resize is None:
            raise ValueError("self.features_3d_resize is None, execute run_3d_model first!")

        idx_in_cropped_patch = slice_idx - self.crop_info["crop_range"][0]

        pos_vectors = {}
        f_3d = {}
        for k in self.features_3d.keys():
            r = int(k[1:])
            pos = idx_in_cropped_patch % r
            pos = torch.diag(torch.ones(r))[pos]
            pos_vectors[k] = pos.to(self.device)

            try:
                f_3d[k] = self.features_3d_resize[k][:,:,idx_in_cropped_patch//r,:,:]
            except:
                f_3d[k] = torch.zeros_like(self.features_3d_resize[k][:,:,0,:,:])
        f_3d, preds = self.f3d_converter.convert_3d_feature(f_3d, pos_vectors, pred_mask=pred)

        D, H, W = self.crop_info["volume_size"]
        for k in f_3d.keys():
            r = int(k[1:])
            if type(self.crop_info["crop_range"])==torch.Tensor:
                self.crop_info["crop_range"] = self.crop_info["crop_range"].numpy()
            assert(np.all(self.crop_info["crop_range"] % r)==0)
            corner = self.crop_info["crop_range"] // r
            b, c, h, w = f_3d[k].shape
            f = torch.zeros(b, c, H//r, W//r).to(self.device)
            f[:,:,corner[1]:corner[4],corner[2]:corner[5]] = f_3d[k]
            f_3d[k] = f

        return f_3d, preds

    def run_int_model(self, image, scribble_mask, prev_round_mask, case_ID = None, slice_idx=None, first_ite = None):
        '''
        input size: (b, 1, H, W)
        scribble_mask: (b, 2, H, W)
        '''
        if case_ID is not None:
            self.current_case = case_ID
        if first_ite is not None:
            self.first_ite = first_ite
        # if self.feature_map_3d is None:
        #     self.feature_map_3d = self.extract_3d_feature(self.volumn)
        if prev_round_mask is None:
            prev_round_mask = torch.ones(image.shape) * 0.5

        if self.gpu:
            image = image.cuda()
            scribble_mask = scribble_mask.cuda()
            prev_round_mask = prev_round_mask.cuda()

        inputs = torch.cat([image, prev_round_mask, scribble_mask], 1) # (b, 4, H, W)
        # if inputs.shape[1] != 4:
        #     print("Unexpected input channel number, expect 4, got ", inputs.shape[1])
        mask = torch.sigmoid(self.int_model(inputs))
        

        return mask

    # def run_prop_model(self, image, segmented_images, segmented_masks, prev_slice_image, prev_slice_mask, prev_round_mask, user_anno_images, user_anno_scribbles):
    def run_prop_model(self, target_image, f_3d, segmented_images, segmented_masks, clear=True, additional_k_v = None):
        # target_image: b*1*h*w, segmented_images, segmented_masks: b*n*h*w, guid_map: b*2*h*w
        # clear: if clear, the stored key and value maps will be deleted; if not clear, the k and v saved before will also be used.

        # if self.feature_map_3d is None:
        #     self.feature_map_3d = self.extract_3d_feature(self.volumn)

        # calculate query key map and value map
        q_k, q_v, q_f16, q_f8, q_f4 = self.prop_model.encode_key(target_image)
        
        if clear:
            self.memory_keys = []
            self.memory_values = []
        
        if len(self.memory_keys)>100:
            self.memory_keys = self.memory_keys[-100:]
            self.memory_values = self.memory_values[-100:]

        # calculate memory key map and value map
        if segmented_images is not None:
            b, k, h, w = segmented_images.shape
            for i in range(k):
                memory_frame = segmented_images[:,i:i+1] # b*1*h*w
                memory_mask = segmented_masks[:,i:i+1]
                m_k, _, m_f16, _, _ = self.prop_model.encode_key(memory_frame) # m_k: b*64*h/16*w/16
                m_v = self.prop_model.encode_value(memory_frame, m_f16, memory_mask) # m_v: b*512*1*h/16*w/16
                self.memory_keys.append(m_k.unsqueeze(2))
                self.memory_values.append(m_v)
            
            memory_keys = torch.cat(self.memory_keys, 2)
            memory_values = torch.cat(self.memory_values, 2) # b*c*t*h*w

            if additional_k_v is not None:
                memory_keys = torch.cat([memory_keys, additional_k_v[0]], 2)
                memory_values = torch.cat([memory_values, additional_k_v[1]], 2)
        else:
            memory_keys = additional_k_v[0]
            memory_values = additional_k_v[1]

        pred_mask, affinity = self.prop_model.segment_with_query(memory_keys, memory_values, q_f8, q_f4, q_k, q_v, f_3d)
        
        return pred_mask, affinity

    def _process_scribble_mask(self, scribble_mask, target_idx, dilate=True):
        if dilate:
            kernel = np.ones((3,3), np.uint8)
            p_srb = (scribble_mask==target_idx).astype(np.uint8)
            p_srb = cv2.dilate(p_srb, kernel).astype(np.bool_)

            n_srb = ((scribble_mask!=target_idx) * (scribble_mask!=-1)).astype(np.uint8)
            n_srb = cv2.dilate(n_srb, kernel).astype(np.bool_)

            Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(self.device)
            Rs, _ = pad_divide_by(Rs, 16, Rs.shape[-2:])
        else:
            p_srb = (scribble_mask==target_idx).astype(np.uint8)
            n_srb = ((scribble_mask!=target_idx) * (scribble_mask!=-1)).astype(np.uint8)
            Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(self.device)
            Rs, _ = pad_divide_by(Rs, 16, Rs.shape[-2:])
        return Rs

    def process_eval_scribble(self, scribble_mask):
        d, h, w = scribble_mask.shape
        idx = np.where(scribble_mask!=-1)[0][0]
        scribble_mask = scribble_mask[idx] # h*w
        
        scribble_mask_new = np.zeros((2,h,w))
        scribble_mask_new[0][np.where(scribble_mask==0)] = 1 # negative
        scribble_mask_new[1][np.where(scribble_mask==1)] = 1
        scribble_mask_new = torch.from_numpy(scribble_mask_new)

        if h < 416:
            scribble_mask = torch.zeros(2, 416, 416)
            scribble_mask[0] = F.interpolate(scribble_mask_new[0:1].unsqueeze(0), size=[416,416], mode="bilinear", align_corners=False)[0,0]
            scribble_mask[1] = F.interpolate(scribble_mask_new[1:].unsqueeze(0), size=[416,416], mode="bilinear", align_corners=False)[0,0]
            scribble_mask[scribble_mask>0.5] = 1
            scribble_mask[scribble_mask<1] = 0
        elif h%16!=0:
            dele, dele2 = self.delete
            scribble_mask = scribble_mask_new[:,dele:-1*dele2,dele:-1*dele2]
        else:
            scribble_mask = scribble_mask_new
        
        _, h, w = scribble_mask.shape
        scribble_3d = torch.zeros(1,2,d,h,w)
        scribble_3d[0,:,idx] = scribble_mask.clone()
        
        return scribble_mask, idx, scribble_3d

    def initialize(self, ct, mask=None, case=None):
        self.volumn = torch.from_numpy(ct) # only ct, without mask
        # print(self.volumn.shape)
        c, h, w = self.volumn.shape
        self.ct_size = self.volumn.shape
        self.delete = None
        if h < 416:
            volumn = self.volumn.unsqueeze(0).unsqueeze(0)
            volumn = F.interpolate(volumn, size=[c,416,416], mode="trilinear", align_corners=False)[0,0]
            self.volumn = volumn
        elif h%16!=0:
            dele = h%16//2
            dele2 = h%16 - dele
            self.volumn = self.volumn[:,dele:-1*dele2,dele:-1*dele2]
            self.delete = [dele,dele2]
        assert(self.volumn.shape[-1]%16==0)
        assert(self.volumn.shape[-1]>=416)
        assert(self.volumn.dim()==3)

        self.pred_masks = torch.zeros_like(self.volumn) # predicted masks
        self.max_idx = self.volumn.shape[0] - 1
        self.segmented_index = []
        self.int_slice_keys = [] # save the memory key maps of each interactive slices
        self.int_slice_values = []
        self.interactive_slice = []
        self.current_round = 0
        self.current_case = case

        self.features_3d = None # feature maps directly predicted by 3d-cnn
        self.pred_patch_mask = None # predicted mask by 3d-cnn
        self.volume_patch = None # cropped and resized 3d patch
        self.features_3d_resize = None # 3d features resized to original cropped size

        if mask is not None:
            self.gt_mask = mask
    
    def get_pred_mask(self, final_pred):
        pred_patch_mask = F.interpolate(self.pred_patch_mask, size=self.crop_info["ori_crop_size"], mode="trilinear", align_corners=False)[0,0]
        pred_mask_seg3d = torch.zeros(self.volumn.shape)
        corners = self.crop_info["crop_range"]
        pred_mask_seg3d[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]] = pred_patch_mask[:corners[3]-corners[0]]

        c, h, w = self.ct_size
        if h < 416:
            pred_mask_seg3d = self.pred_mask_seg3d.unsqueeze(0).unsqueeze(0)
            pred_mask_seg3d = F.interpolate(pred_mask_seg3d, size=[c,h,w], mode="trilinear", align_corners=False)
            pred_mask_seg3d = pred_mask_seg3d.squeeze(0).squeeze(0)
        elif h%16!=0:
            dele, dele2 = self.delete
            pred_3d = torch.zeros(c,h,w)
            pred_3d[:,dele:-1*dele2,dele:-1*dele2] = pred_mask_seg3d
            pred_mask_seg3d = pred_3d
        self.pred_mask_seg3d = pred_mask_seg3d

        if self.only_3d:
            return self.pred_mask_seg3d.numpy()

        if h < 416:
            pred_masks = self.pred_masks.unsqueeze(0).unsqueeze(0)
            pred_masks = F.interpolate(pred_masks, size=[c,h,w], mode="trilinear", align_corners=False)
            pred_masks = pred_masks.squeeze(0).squeeze(0)
        elif h%16!=0:
            dele, dele2 = self.delete
            pred_masks = torch.zeros(c,h,w)
            pred_masks[:,dele:-1*dele2,dele:-1*dele2] = self.pred_masks
        else:
            pred_masks = self.pred_masks

        if final_pred == "prop":
            return pred_masks.numpy()
        elif final_pred == "seg3d":
            return pred_mask_seg3d.numpy()
        elif final_pred == "avg":
            return ((pred_masks+pred_mask_seg3d)/2).numpy()
        else:
            raise ValueError("final_pred not exists, expected ['prop', 'seg3d', 'avg'], got {}".format(final_pred))

    # def array_to_tensor(self, img_list):
    #     for i in range(len(img_list)):
    #         img_list[i] = torch.from_numpy(img_list[i].astype(np.float32))
    #         if self.gpu:
    #             img_list[i] = img_list[i].cuda()
    #     return img_list
    def prepare_tensor(self, img_list):
        for i in range(len(img_list)):
            if type(img_list[i]) == np.ndarray:
                img_list[i] = torch.from_numpy(img_list[i])

            if len(img_list[i].shape)==2:
                img_list[i] = img_list[i].unsqueeze(0).unsqueeze(0)
            elif len(img_list[i].shape)==3:
                img_list[i] = img_list[i].unsqueeze(0)
            
            img_list[i] = img_list[i].float()
            if self.gpu:
                img_list[i] = img_list[i].cuda()
        return img_list
    
    def crop_3d_patch(self, scribble_3d):
        z, h, w = self.volumn.shape
        volume_patch = self.volumn.float()
        scribble_3d = scribble_3d.float()

        if self.pred_patch_mask is None:
            self.pred_patch_mask = torch.zeros(1,1,16,256,256).to(self.device)
        else:
            # pred_patch_mask = torch.zeros(size)
            pred_patch_mask = self.pred_mask_seg3d
            pred_patch_mask = pred_patch_mask.unsqueeze(0).unsqueeze(0)
            self.pred_patch_mask = F.interpolate(pred_patch_mask, size=[16,256,256], mode="trilinear", align_corners=False).to(self.device)
        
        self.crop_info = {"ori_crop_size":volume_patch.shape, "volume_size":self.volumn.shape, "crop_range":torch.tensor([0,0,0,self.volumn.shape[0],self.volumn.shape[1],self.volumn.shape[2]]), "z_padding":0}


        volume_patch = volume_patch.unsqueeze(0).unsqueeze(0)
        self.volume_patch = F.interpolate(volume_patch, size=[16,256,256], mode="trilinear", align_corners=False).to(self.device)
        scribble_3d = F.interpolate(scribble_3d, size=[16,256,256], mode="trilinear", align_corners=False)

        return scribble_3d


def load_model():
    interaction_model = deeplabv3plus_resnet50()
    propagation_model = PropagationNetwork()
    model = iMIS(int_model=interaction_model, prop_model=propagation_model)
    return model
