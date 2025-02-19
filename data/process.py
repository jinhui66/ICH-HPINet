import os
import json
import numpy as np

path = "path"

data_info = {}
for file_name in os.listdir(path):
    if file_name.endswith("npz"):
        print(file_name)
        sample_info = {}
        case_id = file_name.split(".")[0]
        volumn = np.load(os.path.join(path,file_name))["data"]
        seg_mask = volumn[1]
        organ_slice_index = np.unique(np.where(seg_mask==1)[0])
        tumor_slice_index = np.unique(np.where(seg_mask==2)[0])
        volumn_shape = seg_mask.shape
        sample_info = {
            "organ_slice_index":organ_slice_index.tolist(),
            "tumor_slice_index":tumor_slice_index.tolist(),
            "volumn_shape":volumn_shape
        }
        if case_id not in data_info:
            data_info[case_id] = sample_info
        else:
            print("Warning!", case_id)


with open(os.path.join(path,"data_info.json"), "w") as f:
    json.dump(data_info, f) 

