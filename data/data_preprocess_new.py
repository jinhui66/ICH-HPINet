
import os
import pickle
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

crop_size = 96
padding = 10 # if target size is larger than crop_size, padding, crop and resize
ori_data_path = "/data/lyshi/datasets/nnUNet_data/nnUnet_preprocessed/Task050_KiTS/nnUNetData_plans_v2.1_2D_stage0"
save_data_path = "/mntnfs/med_data2/shiluyue/InteractiveCTSeg/dataset/KiTS_Organ_3D_Patch"
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

save_jpg = True
kits_target = "organ" # If processing KiTS19 dataset, choose the target object. ["organ", "tumor", None]

meta = {"crop_size":crop_size}
for f in os.listdir(ori_data_path):
    # if not f.endswith("npz") and not f.endswith("npy"):
    #     continue
    if not f.endswith("npz"):
        continue
    case_id = f.split('.')[0]
    print(case_id)

    volume_path = os.path.join(ori_data_path, f)
    volume = np.load(volume_path)
    if f.endswith("npz"): volume = volume["data"]
    gt_mask = volume[1]
    volume = volume[0]
    z,h,w = volume.shape
    print("volume size:", volume.shape)
    meta[case_id] = {"shape":volume.shape}

    if kits_target=="organ":
        gt_mask[gt_mask!=1] = 0
    elif kits_target=="tumor":
        gt_mask[gt_mask!=2] = 0
        gt_mask[gt_mask==2] = 1

    region = np.where(gt_mask==1)
    corners = np.asarray([np.min(region[i]) for i in range(3)] + [np.max(region[i]) for i in range(3)])
    center = np.asarray([np.mean([corners[i+3], corners[i]]) for i in range(3)])
    lz, lh, lw = [corners[i+3] - corners[i] for i in range(3)]
    crop_length_hw, crop_length_z = [crop_size, crop_size]

    # crop_length_z = min(max(lz + padding, crop_size), z)
    crop_length_z = min(lz+16-lz%16 , z-z%16)
    crop_length_hw = max(max(lh,lw) + padding, crop_size)

    if crop_length_hw%16!=0: 
        crop_length_hw = crop_length_hw + 16 - crop_length_hw%16

    z_padding = 0
    if z-z%16 < lz:
        crop_length_z = z
        z_padding = 16 - z&16

    corners[1:3] = center[1:] - crop_length_hw // 2
    corners[4:6] = corners[1:3] + crop_length_hw
    corners[0] = center[0] - crop_length_z // 2
    corners[3] = corners[0] + crop_length_z
    
    diff = corners[:3] % 16
    corners[:3] -= diff
    corners[3:] -= diff

    for i in range(3):
        l = [z,h,w][i]
        if corners[i] < 0:
            corners[i+3] -= corners[i]
            corners[i] = 0
        if corners[i+3] > l:
            l2 = l - l%16
            corners[i] -= corners[i+3]-l2
            corners[i+3] = l2
    
    assert(np.all(corners>=0))
    assert(corners[3]<=z)
    assert(corners[4]<=h)
    assert(corners[5]<=h)

    cropped_volume = np.zeros((crop_length_z+z_padding, crop_length_hw, crop_length_hw))
    cropped_volume[:crop_length_z] = volume[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]]
    cropped_mask = np.zeros((crop_length_z+z_padding, crop_length_hw, crop_length_hw))
    cropped_mask[:crop_length_z] = gt_mask[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]]

    meta[case_id]["ori_center"] = center
    meta[case_id]["crop_range"] = corners.copy()
    meta[case_id]["crop_size"] = cropped_volume.shape
    meta[case_id]["z_padding"] = z_padding
    print("crop_size:", cropped_volume.shape)

    np.save(os.path.join(save_data_path, "Volume-{}.npy".format(case_id.split('_')[-1])), cropped_volume)
    np.save(os.path.join(save_data_path, "Mask-{}.npy".format(case_id.split('_')[-1])), cropped_mask)

    if save_jpg:
        crop_z = cropped_volume.shape[0]
        plt.figure(figsize=(40,30))
        for i in range(6):
            plt.subplot(3,4,2*i+1)
            plt.imshow(cropped_volume[crop_z//2-3+i], cmap="gray")
            plt.subplot(3,4,2*i+2)
            plt.imshow(cropped_mask[crop_z//2-3+i], cmap="gray")
        if not os.path.exists(os.path.join(save_data_path, "jpg")):
            os.mkdir(os.path.join(save_data_path, "jpg"))
        plt.savefig(os.path.join(save_data_path, "jpg", "{}.jpg".format(case_id.split('_')[-1])))

with open(os.path.join(save_data_path, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)


# instance_count = 0
# meta = {"crop_size":crop_size}
# for f in os.listdir(ori_data_path):
#     # if not f.endswith("npz") and not f.endswith("npy"):
#     #     continue
#     if not f.endswith("npz"):
#         continue
#     case_id = f.split('.')[0]
#     print(case_id, end=' ')

#     volume_path = os.path.join(ori_data_path, f)
#     volume = np.load(volume_path)
#     if f.endswith("npz"): volume = volume["data"]
#     gt_mask = volume[1]
#     volume = volume[0]
#     z,h,w = volume.shape
#     print(volume.shape)
#     meta[case_id] = {"shape":volume.shape}
#     meta[case_id]["instances"] = []

#     if kits_target=="organ":
#         gt_mask[gt_mask!=1] = 0
#     elif kits_target=="tumor":
#         gt_mask[gt_mask!=2] = 0
#         gt_mask[gt_mask==2] = 1

#     instance_mask = measure.label(gt_mask, connectivity=1)
#     instance_idx = []
#     for i in np.unique(instance_mask):
#         if i < 1:
#             continue
#         area = np.sum(instance_mask==i)
#         if area > 100:
#             instance_idx.append(i)
#     meta[case_id]["instance_num"] = len(instance_idx)
#     # max_ins = 2 if target_object=="organ" else 3
#     # if len(instance_idx)>1:
#     #     print(len(instance_idx))
#     # if len(instance_idx) > max_ins:
#     #     print("Warning! Found {} instances in {}.".format(len(instance_idx), case_id))
#     # if self.target_object=="organ" and len(instance_idx)!=2:
#     #     print("Warning! Found only {} organ instances in {}.".format(len(instance_idx), case_id))

#     for ins_i in instance_idx:
#         instance_count += 1
#         ins_mask = instance_mask.copy()
#         ins_mask[instance_mask!=ins_i] = 0
#         ins_mask[instance_mask==ins_i] = 1
#         region = np.where(instance_mask==ins_i)
#         corners = np.asarray([np.min(region[i]) for i in range(3)] + [np.max(region[i]) for i in range(3)])
#         center = np.asarray([np.mean([corners[i+3], corners[i]]) for i in range(3)])
#         lz, lh, lw = [corners[i+3] - corners[i] for i in range(3)]
#         crop_length_hw, crop_length_z = [crop_size, crop_size]

#         crop_length_z = min(max(lz + padding, crop_size), z)
#         crop_length_hw = max(max(lh,lw) + padding, crop_size)

#         corners[1:3] = center[1:] - crop_length_hw // 2
#         corners[4:6] = corners[1:3] + crop_length_hw
#         corners[0] = center[0] - crop_length_z // 2
#         corners[3] = corners[0] + crop_length_z

#         for i in range(3):
#             l = [z,h,w][i]
#             if corners[i] < 0:
#                 corners[i+3] -= corners[i]
#                 corners[i] = 0
#             if corners[i+3] > l:
#                 corners[i] -= corners[i+3]-l
#                 corners[i+3] = l

#         assert(np.all(corners>=0))
#         assert(corners[3]<=z)
#         assert(corners[4]<=h)
#         assert(corners[5]<=h)

#         cropped_volume = volume[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]]
#         cropped_mask = ins_mask[corners[0]:corners[3], corners[1]:corners[4], corners[2]:corners[5]]

#         assert(cropped_volume.shape[0]==crop_length_z)
#         assert(np.all(np.array(cropped_volume.shape[1:])==crop_length_hw))

#         ins_info = {}
#         ins_info["instance_count"] = instance_count
#         ins_info["ori_center"] = center
#         ins_info["crop_range"] = corners.copy()
#         ins_info["crop_size"] = cropped_volume.shape
#         meta[case_id]["instances"].append(ins_info)

#         np.save(os.path.join(save_data_path, "Volume-{}-{}.npy".format(instance_count, case_id.split('_')[-1])), cropped_volume)
#         np.save(os.path.join(save_data_path, "Mask-{}-{}.npy".format(instance_count, case_id.split('_')[-1])), cropped_mask)

#         if save_jpg:
#             crop_z = cropped_volume.shape[0]
#             plt.figure(figsize=(40,30))
#             for i in range(6):
#                 plt.subplot(3,4,2*i+1)
#                 plt.imshow(cropped_volume[crop_z//2-3+i], cmap="gray")
#                 plt.subplot(3,4,2*i+2)
#                 plt.imshow(cropped_mask[crop_z//2-3+i], cmap="gray")
#             if not os.path.exists(os.path.join(save_data_path, "jpg")):
#                 os.mkdir(os.path.join(save_data_path, "jpg"))
#             plt.savefig(os.path.join(save_data_path, "jpg", "{}-{}.jpg".format(instance_count, case_id.split('_')[-1])))

# with open(os.path.join(save_data_path, "meta.pkl"), "wb") as f:
#     pickle.dump(meta, f)
