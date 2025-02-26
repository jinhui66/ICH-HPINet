from genericpath import exists
from time import ctime
import numpy as np
import torch
import random
# import torch.nn.functional as F
from evaluation.interactiveSession import KiTSInteractiveSession
from networks.HPI import HPI
from networks.deepLabV3.modeling import deeplabv3_resnet50, deeplabv3_mobilenet
from networks.STCN.prop_net import PropagationNetwork, Converter
from networks.unet3d.model import ResidualUNet3D

import os
import time
from skimage import io, img_as_ubyte
os.environ['TORCH_HOME'] = './pretrained-model'
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--seg2d_weight', type=str, default="./checkpoints/seg2d_model.pth")
parser.add_argument('--prop_weight', type=str, default="./checkpoints/prop_model.pth")
parser.add_argument('--seg3d_weight', type=str, default="./checkpoints/seg3d_model.pth")
parser.add_argument('--convert_weight', type=str, default="./checkpoints/f3d_converter.pth")
parser.add_argument('--thresh', type=float, default=0.5)
args = parser.parse_args()

print("Loading models")
convert_model = Converter()
pretrained_dict = torch.load(args.convert_weight, map_location=torch.device('cpu'))
convert_model.load_state_dict(pretrained_dict)
convert_model.eval()

seg3d_model = ResidualUNet3D(in_channels=4, out_channels=1, is_segmentation=False, layer_order="bcr")
try:
    pretrained_dict = torch.load(args.seg3d_weight, map_location=torch.device('cpu'))['network']
except:
    pretrained_dict = torch.load(args.seg3d_weight, map_location=torch.device('cpu'))
    
try:
    seg3d_model.load_state_dict(pretrained_dict)
except:
    pretrained_dict = {k[7:]:pretrained_dict[k] for k in pretrained_dict.keys()}
    seg3d_model.load_state_dict(pretrained_dict)
seg3d_model.eval()

pretrained_dict = torch.load(args.seg2d_weight, map_location=torch.device('cpu'))
seg2d_model = deeplabv3_resnet50(num_classes=1, output_stride=16, pretrained_backbone=False)
seg2d_model.load_state_dict(pretrained_dict)

propagation_model = PropagationNetwork()
pretrained_dict = torch.load(args.prop_weight, map_location=torch.device('cpu'))
propagation_model.load_state_dict(pretrained_dict)
propagation_model.eval()

processor = HPI(seg3d_model.cuda(), seg2d_model.cuda(), propagation_model.cuda(), f3d_converter=convert_model.cuda())

save_final_mask = False
save_round_mask = False
data_dir="/data3/wangchangmiao/jinhui/DATA/private_ICH"
ori_data_dir="/data3/wangchangmiao/jinhui/DATA/private_ICH"
val_set='./eval.txt'
shuffle=False
max_time=None
max_nb_interactions=10
metric_to_optimize='Dice'
save_dir = "./eval"
target_object = "ICH"

t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
save_dir = os.path.join(save_dir, t)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_img_dir = os.path.join(save_dir, "img")
save_report_dir = os.path.join(save_dir, "report")

torch.manual_seed(117010231)
np.random.seed(117010231)
random.seed(117010231)

print("Start evaluation...")
with torch.no_grad():

    with KiTSInteractiveSession(data_dir=data_dir,val_set=val_set,shuffle=shuffle,max_time=max_time,max_nb_interactions=max_nb_interactions,
                                metric_to_optimize=metric_to_optimize,save_img_dir=save_img_dir, save_report_dir=save_report_dir, 
                                target_object=target_object, save_results=save_final_mask, ori_data_dir=ori_data_dir, thresh=args.thresh) as sess:
        while sess.next():
            data_index, scribble, first_ite, ct = sess.get_scribbles() # ct, scribble: D*H*W

            if first_ite:
                processor.initialize(ct=ct, case=data_index)
                min_v, max_v = np.min(ct), np.max(ct)
                # print(ct.shape)
                ct2 = (ct - min_v) / (max_v - min_v)
                round_imgs = [[],[],[ct2, sess.gt_mask],[],[]]

            if len(np.unique(scribble))==1:
                print("Warning! No scribble provided.")
                sess.submit_masks(pred_masks)
                continue
            
            # only_int = True if processor.current_round%2==0 else False
            only_int = False
            only_3d = False
            final_pred = "prop"
            if only_3d:
                processor.pred_masks = torch.from_numpy(sess.gt_mask)
                final_pred = "seg3d"
            pred_masks, int_slice = processor.interact(scribble, only_int, final_pred, only_3d)

            if save_round_mask:
                save_path = os.path.join(save_img_dir, "round_result", data_index)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                target_range = np.unique(np.where(pred_masks!=0)[0])
                round_imgs[0].append(np.min(target_range))
                round_imgs[1].append(np.max(target_range))
                round_imgs[2].append(pred_masks)
                round_imgs[3].append(scribble)
                round_imgs[4].append(int_slice)
                if processor.current_round == max_nb_interactions:
                    min_idx = np.min(round_imgs[0])
                    max_idx = np.max(round_imgs[1])
                    z, h, w = ct.shape
                    for j in range(min_idx, max_idx+1):
                        save_img = np.zeros((h, (max_nb_interactions+2)*w))
                        for i in range(max_nb_interactions+2):
                            save_img[:,i*w:(i+1)*w] = round_imgs[2][i][j]
                        io.imsave(os.path.join(save_path, "Slice-{}.jpg".format(j)), img_as_ubyte(save_img), check_contrast=False)
                    save_scrib = np.zeros((h, max_nb_interactions*w, 3)).astype(np.uint8)
                    for i in range(max_nb_interactions):
                        int_slice = round_imgs[4][i]
                        scribble = round_imgs[3][i]
                        save_scrib[:,i*w:(i+1)*w] = np.repeat(round_imgs[2][i+2][int_slice][:,:,np.newaxis]*255, 3, 2).astype(np.uint8)
                        save_scrib[:,i*w:(i+1)*w][scribble[int_slice]==0] = np.array([255,0,0])
                        save_scrib[:,i*w:(i+1)*w][scribble[int_slice]==1] = np.array([0,255,0])
                    # io.imsave(os.path.join(save_path, "Scribble-{}.jpg".format(str(round_imgs[4]))), save_scrib, check_contrast=False)

            if only_int:
                sess.submit_masks(pred_masks, next_scribble_frame_candidates=[int_slice])
            else:
                sess.submit_masks(pred_masks)

