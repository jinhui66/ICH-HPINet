import SimpleITK  as sitk
import os
import numpy as np
import torch

path1 = "/data3/wangchangmiao/jinhui/DATA/private_ICH/binary_mask/"
path2 = "/data3/wangchangmiao/jinhui/DATA/public_ICH/binary_mask/"


output_file1 = open('private_list.txt', 'w')
output_file2 = open('public_list.txt', 'w')
depth = 16


for patient in os.listdir(path1):
    patient_path = path1 + patient
    array = sitk.GetArrayFromImage(sitk.ReadImage(patient_path))
    array = array[len(array)//2 - depth//2 : len(array)//2 + depth//2]
    # print(np.max(array))
    count = np.unique(np.where(array == 1)[0])
    num = np.sum(array > 0.5)
    print(num, np.max(array), array.shape[0], array.shape[1], array.shape[2])
    print(patient, 'true' if len(count) >= 3 else 'false')
    if len(count) >= 3: 
        output_file1.write(f"{patient.split('.')[0]}\n")


print("\n\n\n\n\n")

for patient in os.listdir(path2):
    patient_path = path2 + patient
    array = sitk.GetArrayFromImage(sitk.ReadImage(patient_path))
    tensor = torch.from_numpy(array).type(torch.int16)
    array = tensor.numpy()
    array = array[len(array)//2 - depth//2 : len(array)//2 + depth//2]
    # array = array / 255
    array[array >= 0.5] = 1
    array[array < 0.5] = 0

    # # print(np.max(array))
    count = np.unique(np.where(array==1)[0])
    num = np.sum(array > 0.5)
    print(num, np.max(array), array.shape[0], array.shape[1], array.shape[2])
    # print(count)
    print(np.max(array), array.shape[0], array.shape[1], array.shape[2])
    print(patient, 'true' if len(count) >= 3 else 'false')    
    if len(count) >= 3:
        output_file2.write(f"{patient.split('.')[0]}\n")
        