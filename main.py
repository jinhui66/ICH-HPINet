import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './pretrained-model'

from multiprocessing.dummy import freeze_support
import time
import random
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import RandomVerticalFlip

from config import cfg
from networks.deepLabV3.modeling import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet
from networks.STCN.prop_net import PropagationNetwork, Converter
from networks.unet3d.model import ResidualUNet3D

from networks.HPI import HPI
import torch.optim as optim
from davisinteractive.robot.interactive_robot import InteractiveScribblesRobot
from PIL import Image
from torch.utils.data import DataLoader

import torchvision.transforms as tr
from data.dataloader import ich_Dataset
from networks.loss import Added_BCEWithLogitsLoss, Added_CrossEntropyLoss, DC_and_topk_loss
from utils import AverageMeter, RandomIdentitySampler, PropagationRandomSampler
from collections import defaultdict
from evaluation.metrics import HD, dice, jaccard, MAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainManager(object):
    def __init__(self, use_gpu=True,time_budget=None,
                 
        save_result_dir=cfg.SAVE_RESULT_DIR,pretrained=False,interactive_test=False, save_sample_num = 3):
        
        self.save_sample_num = save_sample_num
        
        if save_result_dir != '':
            self.save_res_dir = save_result_dir
        else:
            self.save_res_dir = save_path

        if not os.path.exists(self.save_res_dir):
            os.makedirs(self.save_res_dir)

        self.seg3d_model = ResidualUNet3D(in_channels=4, out_channels=1, is_segmentation=False, layer_order="bcr").to(device)
        self.seg2d_model = deeplabv3_resnet50(num_classes=1, output_stride=16, pretrained_backbone=False).to(device)
        self.propagation_model = PropagationNetwork().to(device)
        self.f3d_converter = Converter().to(device)
        self.use_gpu=use_gpu
        

        self.HPI = HPI(
            seg3d_model = self.seg3d_model, 
            seg2d_model = self.seg2d_model, 
            prop_model = self.propagation_model,
            f3d_converter = self.f3d_converter 
        )
        
    def train_propagation(self, freeze_params, start_epo=1, loss_func='dice_topk'): #freeze_params=["int3d", "converter", "prop_encoder"]

        
        loss_types = ["loss_convert", "loss_prop", "loss_total", "loss_3d"] 
        running_loss = {}
        for k in loss_types:
            running_loss[k] = AverageMeter()
        
        self.optimizer = optim.Adam([
            {"params":filter(lambda p: p.requires_grad, self.HPI.prop_model.parameters())},
            {"params":filter(lambda p: p.requires_grad, self.HPI.f3d_converter.parameters())}, 
            {"params":filter(lambda p: p.requires_grad, self.HPI.seg3d_model.parameters())}, 
            {"params":filter(lambda p: p.requires_grad, self.HPI.seg2d_model.parameters())}
        ], lr=cfg.TRAIN_PROP_LR, weight_decay=cfg.TRAIN_PROP_WEIGHT_DECAY)

        print('dataset processing...')
        train_dataset = ich_Dataset(preprocessed_data_path=cfg.DATA_ROOT, transform={"flip":0, "crop":0}, train_sample_list=cfg.DATA_TRAIN_LIST, 
                                        object=cfg.TRAIN_OBJECT, data_info_file=cfg.DATA_INFO_FILE)
                
        test_dataset = ich_Dataset(preprocessed_data_path=cfg.DATA_ROOT, transform={"flip":0, "crop":0}, train_sample_list=cfg.DATA_TEST_LIST, 
                                        object=cfg.TRAIN_OBJECT, data_info_file=cfg.DATA_INFO_FILE)
        
        train_volume_num = len(train_dataset.train_id_list)
        test_volumn_num = len(test_dataset.train_id_list)
        print("Length of train/test dataset (volume): ", train_volume_num, " ", test_volumn_num)
        
        trainloader = DataLoader(train_dataset,batch_size=cfg.TRAIN_PROP_BATCH_SIZE,
                sampler = PropagationRandomSampler(train_dataset.train_id_list, num_instances=cfg.TRAIN_PROP_BATCH_SIZE), 
                shuffle=False,num_workers=cfg.TRAIN_PROP_NUM_WORKER,pin_memory=False)
        
        testloader = DataLoader(test_dataset,batch_size=cfg.TRAIN_PROP_BATCH_SIZE,
                sampler = PropagationRandomSampler(test_dataset.train_id_list, num_instances=cfg.TRAIN_PROP_BATCH_SIZE),
                shuffle=False,num_workers=cfg.TRAIN_PROP_NUM_WORKER,pin_memory=False)

        
        # each batch contains b samples which are from the same volume, total batch num equals to the length of sample_list.

        print('dataset processing finished.')
        if loss_func=='bce':
            criterion = Added_BCEWithLogitsLoss(cfg.TRAIN_TOP_K_PERCENT_PIXELS,cfg.TRAIN_HARD_MINING_STEP)
        elif loss_func=='cross_entropy':
            criterion = Added_CrossEntropyLoss(cfg.TRAIN_TOP_K_PERCENT_PIXELS,cfg.TRAIN_HARD_MINING_STEP)
        elif loss_func == "dice_topk":
            criterion = DC_and_topk_loss(ce_kwargs={'k':0})
        else:
            print('unsupported loss funciton. Please choose from [cross_entropy,bce,dice_topk]')

        total_epoch = cfg.TRAIN_PROP_TOTAL_EPO
        dice_score =  AverageMeter()
        hd_score =  AverageMeter()
        jaccard_score =  AverageMeter()
        mae_score = AverageMeter()
        best_dice = 0.0
        
        for epo in range(start_epo, total_epoch+1):
            dice_score.reset()
            hd_score.reset()  
            jaccard_score.reset()
            mae_score.reset()
            print("Train Epo {}: ".format(epo))
            lr=self._adjust_lr(epo, model="PROP")
            start_time = time.time()
            for k in loss_types:
                running_loss[k].reset()

            self.HPI.prop_model.train()
            self.HPI.f3d_converter.train()
            self.HPI.seg3d_model.train()
            self.HPI.seg2d_model.train()

            for ii, sample in enumerate(trainloader):
                if ii >= train_volume_num:
                    print("Warning! The batch num is larger than sample volume num, check dataloader sampler.")
                start_slice = sample["start_slice"].float()
                start_slice_mask = sample["start_slice_mask"].float()
                slice_scribble = sample["slice_scribble"].float()
                slice_prev_mask = sample['slice_prev_mask'].float()
                slice_1 = sample["slice_1"].float()
                slice_1_mask = sample["slice_1_mask"].float()
                slice_2 = sample["slice_2"].float()
                slice_2_mask = sample["slice_2_mask"].float()
                caseID = sample["meta"]["caseID"]
                slice_idx = sample["meta"]["slice_idx"]
                volume_patch = sample["volume_patch"][0:1].float() # 1*1*D*H*W
                mask_patch = sample["mask_patch"][0:1].float()
                prev_round_mask = sample["prev_round_mask"][0:1].float()
                scribble_mask = sample["scribble_mask"][0:1].float()
                crop_info = sample["crop_info"]
                for k in crop_info.keys():
                    crop_info[k] = crop_info[k][0]
                bs,_,h,w = start_slice.size()
                
                if np.unique(np.array(caseID)).shape[0] != 1:
                    print("Warning! Not all slices in the batch are from the same volume, check dataloader sampler.")

                if self.use_gpu:
                    start_slice = start_slice.to(device)
                    start_slice_mask = start_slice_mask.to(device)
                    slice_scribble = slice_scribble.to(device)
                    slice_prev_mask = slice_prev_mask.to(device)
                    slice_1 = slice_1.to(device)
                    slice_1_mask = slice_1_mask.to(device)
                    slice_2 = slice_2.to(device)
                    slice_2_mask = slice_2_mask.to(device)
                    volume_patch = volume_patch.to(device)
                    mask_patch = mask_patch.to(device)
                    prev_round_mask = prev_round_mask.to(device)
                    scribble_mask = scribble_mask.to(device)

                self.HPI.volume_patch = volume_patch
                self.HPI.pred_patch_mask = prev_round_mask
                self.HPI.crop_info = crop_info
                
                if "int3d" in freeze_params:
                    with torch.no_grad():
                        _ = self.HPI.run_3d_model(scribble_mask, crop_first=False)
                    loss_3d = 0
                else:
                    pred_patch_mask = self.HPI.run_3d_model(scribble_mask, crop_first=False)
                    _, _, loss_3d = criterion(net_output=pred_patch_mask, target=mask_patch)

                # print(slice_idx[0][0])
                # print(start_slice.shape, slice_scribble.shape,slice_prev_mask.shape,slice_idx)
                pred_slice_mask = self.HPI.run_2d_model(start_slice, slice_scribble, slice_prev_mask, case_ID = caseID, slice_idx=slice_idx)
                # print(pred_slice_mask.shape, start_slice_mask.shape)
                loss_convert, loss_convert_ce, loss_convert_dc = 0,0,0
                f_3d_dict = [defaultdict(list), defaultdict(list)]
                crop = crop_info["crop_range"]
                target_mask = [slice_1_mask[:,:,crop[1]:crop[4],crop[2]:crop[5]], slice_2_mask[:,:,crop[1]:crop[4],crop[2]:crop[5]]]
                convert_preds = [defaultdict(list), defaultdict(list)]
                for i in range(bs):
                    for j in range(2):
                        f_3d, preds = self.HPI.decode_3d_feature(slice_idx[j+1][i], pred=True)
                        for k in f_3d.keys():
                            f_3d_dict[j][k].append(f_3d[k])
                            if k in preds.keys():
                                convert_preds[j][k].append(preds[k])
                for i in range(2):
                    for k in f_3d.keys():
                        f_3d_dict[i][k] = torch.cat(f_3d_dict[i][k], dim=0).contiguous()
                        if k in preds.keys():
                            convert_preds[i][k] = torch.cat(convert_preds[i][k], dim=0).contiguous()
                            if "converter" not in freeze_params:
                                l1, l2, l3 = criterion(net_output=convert_preds[i][k], target=target_mask[i].contiguous())
                                loss_convert += l1
                                loss_convert_ce += l2
                                loss_convert_dc += l3
                loss_convert /= 4
                loss_convert_ce /= 4
                loss_convert_dc /= 4

                pred_slice_1_mask, _ = self.HPI.run_prop_model(target_image=slice_1, f_3d=f_3d_dict[0], \
                    segmented_images=start_slice, segmented_masks=pred_slice_mask, clear=True)
                pred_slice_2_mask, _ = self.HPI.run_prop_model(target_image=slice_2, f_3d=f_3d_dict[1], \
                    segmented_images=slice_1, segmented_masks=pred_slice_1_mask, clear=True)

                loss_0, ce_loss_0, dc_loss_0 = criterion(net_output=pred_slice_mask,target=start_slice_mask)
                loss_1, ce_loss_1, dc_loss_1 = criterion(net_output=pred_slice_1_mask,target=slice_1_mask)
                loss_2, ce_loss_2, dc_loss_2 = criterion(net_output=pred_slice_2_mask,target=slice_2_mask)

                loss_prop = (loss_0 + loss_1 + loss_2) / 3
                loss_prop_ce = (ce_loss_1 + ce_loss_2) / 2
                loss_prop_dc = (dc_loss_1 + dc_loss_2) / 2
                # print(loss_prop, loss_convert, loss_3d)
                loss_total = 5 * loss_prop + loss_convert + 4*loss_3d
                # loss_total = 5 * loss_prop + 4*loss_3d
                # print(loss_total)

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                for k in loss_types:
                    try:
                        running_loss[k].update(eval(k).detach().cpu().item())
                    except:
                        running_loss[k].update(eval(k))
                
                pred_slice_1_mask[pred_slice_1_mask >= 0.5] = 1
                pred_slice_1_mask[pred_slice_1_mask < 0.5] = 0
                pred_slice_2_mask[pred_slice_2_mask >= 0.5] = 1
                pred_slice_2_mask[pred_slice_2_mask < 0.5] = 0
                
                for b in range(bs):
                
                    # print(slice_1_mask.shape, pred_slice_1_mask.shape)
                    slice_1_dice = dice(slice_1_mask[b].cpu(), pred_slice_1_mask[b].cpu()).item()
                    slice_2_dice = dice(slice_2_mask[b].cpu(), pred_slice_2_mask[b].cpu()).item()
                    # print(torch.max(pred_slice_1_mask), torch.min(pred_slice_1_mask))
                    dice_score.update(slice_1_dice, 1)
                    dice_score.update(slice_2_dice, 1)
                    
                    slice_1_jaccard = jaccard(slice_1_mask[b].cpu(), pred_slice_1_mask[b].cpu()).item()
                    slice_2_jaccard = jaccard(slice_2_mask[b].cpu(), pred_slice_2_mask[b].cpu()).item()
                    jaccard_score.update(slice_1_jaccard, 1)
                    jaccard_score.update(slice_2_jaccard, 1)
                    
                    slice_1_mae = MAE(slice_1_mask[b].cpu(), pred_slice_1_mask[b].cpu()).item()
                    slice_2_mae = MAE(slice_2_mask[b].cpu(), pred_slice_2_mask[b].cpu()).item()
                    mae_score.update(slice_1_mae, 1)
                    mae_score.update(slice_2_mae, 1)

                    # 计算Hausdorff距离
                    hd_1 = HD(slice_1_mask[b][0].cpu(), pred_slice_1_mask[b][0].cpu())
                    hd_2 = HD(slice_2_mask[b][0].cpu(), pred_slice_2_mask[b][0].cpu())
                    # print(hd_1, hd_2)
                    if not math.isinf(hd_1):
                        hd_score.update(hd_1, 1)
                    if not math.isinf(hd_2):
                        hd_score.update(hd_2, 1)
                
            dice_slice = dice_score.avg
            hd_slice = hd_score.avg  # 计算平均Hausdorff距离
            jaccard_slice = jaccard_score.avg
            mae_slice = mae_score.avg
            
            end_time = time.time()
            print("time-{}, Lr-{}".format(end_time-start_time, lr))
            print("train Loss:", [(k, running_loss[k].avg) for k in loss_types])
            print("train Dice_slice:", dice_slice)
            print("train HD_slice:", hd_slice)
            print("train Jaccard_slice:", jaccard_slice)
            print("train MAE_slice:", mae_slice)

            dice_score.reset()
            hd_score.reset()  
            jaccard_score.reset()
            mae_score.reset()
            for k in loss_types:
                running_loss[k].reset()

            self.HPI.prop_model.eval()
            self.HPI.f3d_converter.eval()
            self.HPI.seg3d_model.eval()
            self.HPI.seg2d_model.eval()
            
            with torch.no_grad():
                for ii, sample in enumerate(testloader):
                    start_slice = sample["start_slice"].float()
                    start_slice_mask = sample["start_slice_mask"].float()
                    slice_scribble = sample["slice_scribble"].float()
                    slice_prev_mask = sample['slice_prev_mask'].float()
                    slice_1 = sample["slice_1"].float()
                    slice_1_mask = sample["slice_1_mask"].float()
                    slice_2 = sample["slice_2"].float()
                    slice_2_mask = sample["slice_2_mask"].float()
                    caseID = sample["meta"]["caseID"]
                    slice_idx = sample["meta"]["slice_idx"]
                    volume_patch = sample["volume_patch"][0:1].float() # 1*1*D*H*W
                    mask_patch = sample["mask_patch"][0:1].float()
                    prev_round_mask = sample["prev_round_mask"][0:1].float()
                    scribble_mask = sample["scribble_mask"][0:1].float()
                    crop_info = sample["crop_info"]
                    for k in crop_info.keys():
                        crop_info[k] = crop_info[k][0]
                    bs,_,h,w = start_slice.size()
                    
                    if np.unique(np.array(caseID)).shape[0] != 1:
                        print("Warning! Not all slices in the batch are from the same volume, check dataloader sampler.")

                    if self.use_gpu:
                        start_slice = start_slice.to(device)
                        start_slice_mask = start_slice_mask.to(device)
                        slice_scribble = slice_scribble.to(device)
                        slice_prev_mask = slice_prev_mask.to(device)
                        slice_1 = slice_1.to(device)
                        slice_1_mask = slice_1_mask.to(device)
                        slice_2 = slice_2.to(device)
                        slice_2_mask = slice_2_mask.to(device)
                        volume_patch = volume_patch.to(device)
                        mask_patch = mask_patch.to(device)
                        prev_round_mask = prev_round_mask.to(device)
                        scribble_mask = scribble_mask.to(device)

                    self.HPI.volume_patch = volume_patch
                    self.HPI.pred_patch_mask = prev_round_mask
                    self.HPI.crop_info = crop_info
                    
                    pred_patch_mask = self.HPI.run_3d_model(scribble_mask, crop_first=False)
                    _, _, loss_3d = criterion(net_output=pred_patch_mask, target=mask_patch)

                    pred_slice_mask = self.HPI.run_2d_model(start_slice, slice_scribble, slice_prev_mask, case_ID = caseID, slice_idx=slice_idx)
                    
                    # pred_patch_mask = self.HPI.run_3d_model(scribble_mask, crop_first=False)

                    f_3d_dict = [defaultdict(list), defaultdict(list)]
                    crop = crop_info["crop_range"]
                    target_mask = [slice_1_mask[:,:,crop[1]:crop[4],crop[2]:crop[5]], slice_2_mask[:,:,crop[1]:crop[4],crop[2]:crop[5]]]
                    for i in range(bs):
                        for j in range(2):
                            f_3d, preds = self.HPI.decode_3d_feature(slice_idx[j+1][i], pred=True)
                            for k in f_3d.keys():
                                f_3d_dict[j][k].append(f_3d[k])
                                
                    for i in range(2):
                        for k in f_3d.keys():
                            f_3d_dict[i][k] = torch.cat(f_3d_dict[i][k], dim=0).contiguous()

                    loss_convert /= 4
                    loss_convert_ce /= 4
                    loss_convert_dc /= 4

                    pred_slice_1_mask, _ = self.HPI.run_prop_model(target_image=slice_1, f_3d=f_3d_dict[0], \
                        segmented_images=start_slice, segmented_masks=pred_slice_mask, clear=True)
                    pred_slice_2_mask, _ = self.HPI.run_prop_model(target_image=slice_2, f_3d=f_3d_dict[1], \
                        segmented_images=slice_1, segmented_masks=pred_slice_1_mask, clear=True)
                    
                    pred_slice_1_mask[pred_slice_1_mask >= 0.5] = 1
                    pred_slice_1_mask[pred_slice_1_mask < 0.5] = 0
                    pred_slice_2_mask[pred_slice_2_mask >= 0.5] = 1
                    pred_slice_2_mask[pred_slice_2_mask < 0.5] = 0
                    
                    # 计算Dice系数
                    for b in range(bs):
                        
                        slice_1_dice = dice(slice_1_mask[b].cpu(), pred_slice_1_mask[b].cpu()).item()
                        slice_2_dice = dice(slice_2_mask[b].cpu(), pred_slice_2_mask[b].cpu()).item()
                        # print(torch.max(pred_slice_1_mask), torch.min(pred_slice_1_mask))
                        dice_score.update(slice_1_dice, 1)
                        dice_score.update(slice_2_dice, 1)
                        
                        slice_1_jaccard = jaccard(slice_1_mask[b].cpu(), pred_slice_1_mask[b].cpu()).item()
                        slice_2_jaccard = jaccard(slice_2_mask[b].cpu(), pred_slice_2_mask[b].cpu()).item()
                        jaccard_score.update(slice_1_jaccard, 1)
                        jaccard_score.update(slice_2_jaccard, 1)
                        
                        slice_1_mae = MAE(slice_1_mask[b].cpu(), pred_slice_1_mask[b].cpu()).item()
                        slice_2_mae = MAE(slice_2_mask[b].cpu(), pred_slice_2_mask[b].cpu()).item()
                        mae_score.update(slice_1_mae, 1)
                        mae_score.update(slice_2_mae, 1)

                        # 计算Hausdorff距离
                        hd_1 = HD(slice_1_mask[b][0].cpu(), pred_slice_1_mask[b][0].cpu())
                        hd_2 = HD(slice_2_mask[b][0].cpu(), pred_slice_2_mask[b][0].cpu())
                        # print(hd_1, hd_2)
                        if not math.isinf(hd_1):
                            hd_score.update(hd_1, 1)
                        if not math.isinf(hd_2):
                            hd_score.update(hd_2, 1)
                        
                dice_slice = dice_score.avg
                hd_slice = hd_score.avg  # 计算平均Hausdorff距离
                jaccard_slice = jaccard_score.avg
                mae_slice = mae_score.avg

                end_time = time.time()

                print("test Loss:", [(k, running_loss[k].avg) for k in loss_types])
                print("test Dice_slice:", dice_slice)
                print("test HD_slice:", hd_slice)
                print("test Jaccard_slice:", jaccard_slice)
                print("test MAE_slice:", mae_slice)
            
                if dice_slice >= best_dice:
                    best_dice = dice_slice
                    torch.save(self.HPI.prop_model.state_dict(), './checkpoints/private/prop_model.pth')
                    torch.save(self.HPI.f3d_converter.state_dict(), './checkpoints/private/f3d_converter.pth')
                    torch.save(self.HPI.seg3d_model.state_dict(), './checkpoints/private/seg3d_model.pth')
                    torch.save(self.HPI.seg2d_model.state_dict(), './checkpoints/private/seg2d_model.pth')

    def load_network(self,net,pretrained_dict):
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        return net

    def _adjust_lr(self, epo, model="PROP"):
        # now_lr = cfg.TRAIN_INT_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_INT_POWER
        # optimizer.param_groups[0]['lr'] = now_lr
        if model=="INT":
            initial_lr = cfg.TRAIN_INT_LR
            steps = cfg.TRAIN_INT_LR_STEP
            gamma = cfg.TRAIN_INT_LR_GAMMA
        elif model=="PROP":
            initial_lr = cfg.TRAIN_PROP_LR
            steps = cfg.TRAIN_PROP_LR_STEP
            gamma = cfg.TRAIN_PROP_LR_GAMMA
        if steps == []:
            lr = initial_lr
        else:
            steps = np.sort(steps)
            index = np.searchsorted(steps, epo)
            lr = initial_lr * np.power(gamma, index)
        self.optimizer.param_groups[0]['lr'] = lr
        return lr

    def load_deeplab(self, path):
        cur_dict = self.interaction_model.state_dict()
        src_dict = torch.load(path, map_location=torch.device('cpu'))['model_state']

        for k in list(src_dict.keys()):
            if type(src_dict[k]) is not int:
                if src_dict[k].shape != cur_dict[k].shape:
                    # print('Reloading: ', k)
                    if 'bias' in k:
                        # Reseting the class prob bias
                        src_dict[k] = torch.zeros_like((src_dict[k][0:1]))
                    elif src_dict[k].shape[1] != 3:
                        # Reseting the class prob weight
                        src_dict[k] = torch.zeros_like((src_dict[k][0:1]))
                        nn.init.orthogonal_(src_dict[k])
                    else:
                        # sum over the weights of input layer to use for grayscale images
                        pads_1 = src_dict[k].sum(dim=1, keepdim=True)
                        # Adding the mask and scribbles channel
                        pads = torch.zeros((64,3,7,7), device=src_dict[k].device)
                        nn.init.orthogonal_(pads)
                        src_dict[k] = torch.cat([pads_1, pads], 1)

        self.interaction_model.load_state_dict(src_dict)

cfg.TRAIN_OBJECT = "ICH"
cfg.SAVE_LOG_DIR = "./log"
if cfg.SAVE_LOG:
    from tensorboardX import SummaryWriter
    # if not os.path.exists(cfg.SAVE_LOG_DIR):
    #     os.makedirs(cfg.SAVE_LOG_DIR)
    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
    save_path = os.path.join(cfg.SAVE_LOG_DIR, start_time)
    logger = SummaryWriter(save_path)

torch.manual_seed(888)
np.random.seed(888)
random.seed(888)

cfg.TRAIN_SEG3D_LR = None
cfg.TRAIN_CONVERT_LR = None

cfg.TRAIN_PROP_BATCH_SIZE = 1
cfg.TRAIN_PROP_SAVE_MODEL_INTERVAL = 20
cfg.PRETRAINED_MODEL_3D = ""
cfg.TRAIN_PROP_TOTAL_EPO = 500
cfg.TRAIN_PROP_LR = 1e-5
cfg.TRAIN_PROP_LR_STEP = []
cfg.TRAIN_PROP_LR_GAMMA = 0.5
cfg.TRAIN_PROP_WEIGHT_DECAY = 1e-7
cfg.PRETRAINED_MODEL_PROP = ""
cfg.PRETRAINED_MODEL_CONVERT = ""
train_manager = TrainManager()
# train_manager.HPI.f3d_converter.load_state_dict(converter_weight)
train_manager.train_propagation(start_epo=1, freeze_params='')
