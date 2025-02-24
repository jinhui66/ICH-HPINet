import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './pretrained-model'

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.serialization import check_module_version_greater_or_equal
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import RandomVerticalFlip

from config import cfg
from networks.deepLabV3.modeling import deeplabv3plus_resnet50, deeplabv3_mobilenet, deeplabv3_resnet50
from networks.STCN.prop_net import PropagationNetwork, Converter
from networks.unet3d.model import ResidualUNet3D

from networks.HPI import HPI
import torch.optim as optim
from davisinteractive.robot.interactive_robot import InteractiveScribblesRobot
from PIL import Image
from torch.utils.data import DataLoader

import torchvision.transforms as tr
from data.dataloader import ich_Dataset
from networks.loss import DC_and_topk_loss
from utils import AverageMeter, TestIdentitySampler, PropagationRandomSampler

from tensorboardX import SummaryWriter
from evaluation.metrics import dice
from collections import defaultdict
from evaluation.metrics import HD, dice
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestManager(object):
    def __init__(self, save_result_dir, trained_weights={"seg3d":'', "convert":'', "seg2d":'', "prop":''}, use_gpu=True):
        '''
        trained_weights: {"seg3d":path, "seg2d":path, "prop":path, "convert":path}
        '''

        self.save_res_dir = save_result_dir
        self.weights = trained_weights
        self.object = cfg.TRAIN_OBJECT
        self.int_batch_size = 12 if self.object=="organ" else 6
        self.prop_batch_size = 3

        if not os.path.exists(self.save_res_dir):
            os.makedirs(self.save_res_dir)

        seg3d_model = ResidualUNet3D(in_channels=4, out_channels=1, is_segmentation=False, layer_order="bcr").to(device)
        interaction_model = deeplabv3_resnet50(num_classes=1, output_stride=16, pretrained_backbone=False).to(device)
        propagation_model = PropagationNetwork().to(device)
        convert_model = Converter().to(device)

        self.use_gpu=use_gpu
        self.device = device
        if self.use_gpu:
            seg3d_model = seg3d_model.to(device)
            interaction_model = interaction_model.to(device)
            propagation_model = propagation_model.to(device)
            convert_model = convert_model.to(device)

        self.HPI = HPI(
            seg3d_model = seg3d_model, 
            int_model = interaction_model, 
            prop_model = propagation_model, 
            f3d_converter=convert_model
        )

        self.load_weights()

    def load_weights(self):
        self.trained_models = []
        if self.weights["seg3d"] != '':
            pretrained_dict = torch.load(self.weights["seg3d"], map_location=torch.device(self.device))
            self.HPI.seg3d_model.load_state_dict(pretrained_dict)

        if self.weights["prop"] != '':
            pretrained_dict = torch.load(self.weights["prop"], map_location=torch.device(self.device))
            self.HPI.prop_model.load_state_dict(pretrained_dict)
            
        if self.weights["convert"] != '':
            pretrained_dict = torch.load(self.weights["convert"], map_location=torch.device(self.device))
            self.HPI.f3d_converter.load_state_dict(pretrained_dict)

        if self.weights["seg2d"] != '':
            pretrained_dict = torch.load(self.weights["seg2d"], map_location=torch.device(self.device))
            self.HPI.int_model.load_state_dict(pretrained_dict)
            

    def test_single_propagation(self, loss_func='dice_topk', step=1, max_step=None, logger=None):
        start_time = time.time()

        self.HPI.prop_model.eval()

        loss_types = ["loss_convert", "loss_convert_ce", "loss_convert_dc", "loss_prop", "loss_prop_ce", "loss_prop_dc", "loss_total", "loss_3d"] 
        running_loss = {}
        for k in loss_types:
            running_loss[k] = AverageMeter()
        dice_score =  AverageMeter()
        hd_score =  AverageMeter()
        dice_score.reset()
        hd_score.reset()

        composed_transforms = None #[tr.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
        train_dataset = ich_Dataset(preprocessed_data_path=cfg.DATA_ROOT, transform={"flip":0, "crop":0}, train_sample_list=cfg.DATA_TRAIN_LIST, 
                                        object=cfg.TRAIN_OBJECT, data_info_file=cfg.DATA_INFO_FILE)
        # if max_step is None:
        #     train_dataset.process = 0.5
        # else:
        #     train_dataset.process = step / max_step
        train_dataset.process = 1
        testloader = DataLoader(train_dataset,batch_size=self.prop_batch_size,
                sampler = PropagationRandomSampler(train_dataset.train_id_list, num_instances=self.prop_batch_size), 
                shuffle=False,num_workers=cfg.TRAIN_PROP_NUM_WORKER,pin_memory=True)

        if loss_func == "dice_topk":
            criterion = DC_and_topk_loss(ce_kwargs={'k':10})
        else:
            print('unsupported loss funciton. Please choose from [dice_topk]')
        hd_slices = []
        ba = 0
        for ii, sample in enumerate(testloader):
            # print("{}/{}".format(ii+1, len(testloader)))
            start_slice = sample["start_slice"].float()
            start_slice_mask = sample["start_slice_mask"].float()
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
                start_slice = start_slice.cuda()
                start_slice_mask = start_slice_mask.cuda()
                slice_1 = slice_1.cuda()
                slice_1_mask = slice_1_mask.cuda()
                slice_2 = slice_2.cuda()
                slice_2_mask = slice_2_mask.cuda()
                volume_patch = volume_patch.cuda()
                mask_patch = mask_patch.cuda()
                prev_round_mask = prev_round_mask.cuda()
                scribble_mask = scribble_mask.cuda()

            self.HPI.volume_patch = volume_patch
            self.HPI.pred_patch_mask = prev_round_mask
            self.HPI.crop_info = crop_info

            pred_patch_mask = self.HPI.run_3d_model(scribble_mask, crop_first=False)
            _, _, loss_3d = criterion(net_output=pred_patch_mask, target=mask_patch)

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
                        l1, l2, l3 = criterion(net_output=convert_preds[i][k], target=target_mask[i].contiguous())
                        loss_convert += l1
                        loss_convert_ce += l2
                        loss_convert_dc += l3
            loss_convert /= 4
            loss_convert_ce /= 4
            loss_convert_dc /= 4

            pred_slice_1_mask, _ = self.HPI.run_prop_model(target_image=slice_1, f_3d=f_3d_dict[0], \
                segmented_images=start_slice, segmented_masks=start_slice_mask, clear=True)
            pred_slice_2_mask, _ = self.HPI.run_prop_model(target_image=slice_2, f_3d=f_3d_dict[1], \
                segmented_images=slice_1, segmented_masks=pred_slice_1_mask, clear=False)
            
            loss_1, ce_loss_1, dc_loss_1 = criterion(net_output=pred_slice_1_mask,target=slice_1_mask)
            loss_2, ce_loss_2, dc_loss_2 = criterion(net_output=pred_slice_2_mask,target=slice_2_mask)

            loss_prop = (loss_1 + loss_2) / 2
            loss_prop_ce = (ce_loss_1 + ce_loss_2) / 2
            loss_prop_dc = (dc_loss_1 + dc_loss_2) / 2
            # print(loss_prop, loss_convert, loss_3d)
            loss_total = 10 * loss_prop + loss_convert + 4*loss_3d
            # print(loss_total)
            for k in loss_types:
                try:
                    running_loss[k].update(eval(k).detach().cpu().item())
                except:
                    running_loss[k].update(eval(k))
            
            pred_slice_1_mask[pred_slice_1_mask >= 0.5] = 1
            pred_slice_1_mask[pred_slice_1_mask < 0.5] = 0
            pred_slice_2_mask[pred_slice_2_mask >= 0.5] = 1
            pred_slice_2_mask[pred_slice_2_mask < 0.5] = 0
            
            image1 = slice_1.cpu().numpy()
            image2 = slice_2.cpu().numpy()
            pred1 = pred_slice_1_mask.cpu().numpy()
            pred2 = pred_slice_2_mask.cpu().numpy()
            mask1 = slice_1_mask.cpu().numpy()
            mask2 = slice_2_mask.cpu().numpy()
            
            ba += 1
            
            # plt.axis('off')
            
            # image = Image.fromarray(image1[0][0].astype(np.uint8))
            # image.save(f"./visual/{ba}_{caseID[0]}_up_input.png")

            # mask1 = mask1[0][0].astype(np.uint8)*255
            # red_channel_image = np.stack((mask1, np.zeros_like(mask1), np.zeros_like(mask1)), axis=-1)
            # image = Image.fromarray(red_channel_image)
            # image.save(f"./visual/{ba}_{caseID[0]}_up_mask.png")

            # image = Image.fromarray(pred1[0][0].astype(np.uint8)*255)
            # image.save(f"./visual/{ba}_{caseID[0]}_up_pred.png")

            # image = Image.fromarray(image2[0][0].astype(np.uint8))
            # image.save(f"./visual/{ba}_{caseID[0]}_down_input.png")
            
            # mask2 = mask2[0][0].astype(np.uint8)*255
            # red_channel_image = np.stack((mask2, np.zeros_like(mask2), np.zeros_like(mask2)), axis=-1)
            # image = Image.fromarray(red_channel_image)
            # image.save(f"./visual/{ba}_{caseID[0]}_down_mask.png")

            # image = Image.fromarray(pred2[0][0].astype(np.uint8)*255)
            # image.save(f"./visual/{ba}_{caseID[0]}_down_pred.png")        
            
            # 计算Dice系数
            for b in range(bs):
                
                # print(slice_1_mask.shape, pred_slice_1_mask.shape)
                
                slice_1_dice = dice(slice_1_mask[b].cpu(), pred_slice_1_mask[b].cpu()).item()
                slice_2_dice = dice(slice_2_mask[b].cpu(), pred_slice_2_mask[b].cpu()).item()
                # print(torch.max(pred_slice_1_mask), torch.min(pred_slice_1_mask))
                dice_score.update(slice_1_dice, 1)
                dice_score.update(slice_2_dice, 1)

                # 计算Hausdorff距离
                hd_1 = HD(slice_1_mask[b][0].cpu(), pred_slice_1_mask[b][0].cpu())
                hd_2 = HD(slice_2_mask[b][0].cpu(), pred_slice_2_mask[b][0].cpu())
                hd_score.update(hd_1, 1)
                hd_score.update(hd_2, 1)
                
            # print(dice_score, hd_slices)
            # print(dice_score.avg)
            # print(hd_slices)
            
        dice_slice = dice_score.avg
        hd_slice = hd_score.avg  # 计算平均Hausdorff距离

        end_time = time.time()

        print("Step-{}, time-{}".format(step, end_time-start_time))
        print("Loss:", [(k, running_loss[k].avg) for k in loss_types])
        print("HD_slice:", hd_slice)
        print("Dice_slice:", dice_slice)
        if logger is not None:
            for k in loss_types:
                logger.add_scalar(k, running_loss[k].avg, step)
            logger.add_scalar("dice_slice", dice_score.avg, step)
            logger.add_text("case_id", caseID[0], step)
            logger.add_text("slice_idx", "{}-{}-{}".format(slice_idx[0][0],slice_idx[1][0],slice_idx[2][0]), step)
            logger.add_image("memory_img", start_slice[0].cpu(), step)
            logger.add_image("memory_mask", start_slice_mask[0].cpu(), step)
            logger.add_image("slice_1_img", slice_1[0].cpu(), step)
            logger.add_image("slice_1_gt_mask", slice_1_mask[0].cpu(), step)
            logger.add_image("slice_1_pred_mask", pred_slice_1_mask[0].detach().cpu(), step)
            logger.add_image("slice_2_img", slice_2[0].cpu(), step)
            logger.add_image("slice_2_gt_mask", slice_2_mask[0].cpu(), step)
            logger.add_image("slice_2_pred_mask", pred_slice_2_mask[0].detach().cpu(), step)

    def test_multi_interaction(self, int_weights, init_seg=False):
        start_time = time.strftime("%Y%m%d_%H%M", time.localtime()) 
        save_path = os.path.join(self.save_res_dir, "test_int_init-{}_{}".format(init_seg,start_time))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger = SummaryWriter(save_path)

        for weight_path in int_weights:
            print(os.path.basename(weight_path))
            self.weights["seg2d"] = weight_path
            self.load_weights()

            step = int(weight_path.split("_")[-1].split('.')[0][3:])
            self.test_single_interaction(step=step, init_seg=init_seg, logger=logger)
            logger.add_text("weight_path", weight_path, step)

    def test_multi_propagation(self, prop_weights, converter_weights, seg3d_weights=None):
        start_time = time.strftime("%Y%m%d_%H%M", time.localtime()) 
        save_path = os.path.join(self.save_res_dir, "test_prop_{}".format(start_time))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger = SummaryWriter(save_path)


        print(os.path.basename(prop_weights), os.path.basename(converter_weights))
        self.weights["prop"] = prop_weights
        self.weights["convert"] = converter_weights
        if seg3d_weights is not None:
            self.weights["seg3d"] = seg3d_weights
        self.load_weights()

        self.test_single_propagation(logger=logger)
        # logger.add_text("weight_path", weight_path, step)


torch.set_grad_enabled(False)
torch.manual_seed(117010231)
np.random.seed(117010231)
random.seed(117010231)

cfg.TRAIN_OBJECT = "organ" 
result_dir = "./result" 

p = result_dir
prop_weights = "/data3/wangchangmiao/jinhui/ICH-HPINet/checkpoints/private/prop_model.pth"
convert_weights = "/data3/wangchangmiao/jinhui/ICH-HPINet/checkpoints/private/f3d_converter.pth"
seg3d_weights = "/data3/wangchangmiao/jinhui/ICH-HPINet/checkpoints/private/seg3d_model.pth"
seg2d_weights = "/data3/wangchangmiao/jinhui/ICH-HPINet/checkpoints/private/seg2d_model.pth"

test_manager = TestManager(
    save_result_dir = result_dir,
    trained_weights = {
        "seg3d":seg3d_weights, 
        "convert":convert_weights, "seg_2d":seg2d_weights, "prop":prop_weights
    }
)

test_manager.test_multi_propagation(prop_weights, convert_weights, seg3d_weights=seg3d_weights)
# int_weights = [os.path.join(p,"int_model_epo0%04d.pth"%(50*i)) for i in range(11,15)]
# test_manager.test_multi_interaction(int_weights, init_seg=False)
