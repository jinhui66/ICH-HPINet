import SimpleITK  as sitk
import os
import numpy as np
import torch

path1 = "../DATA/private_ICH/binary_mask/"
path2 = "../DATA/public_ICH/binary_mask/"


output_file1 = open('train.txt', 'w')
output_file2 = open('public_train.txt', 'w')

for patient in os.listdir(path1):
    patient_path = path1 + patient
    array = sitk.GetArrayFromImage(sitk.ReadImage(patient_path))
    array = array[len(array)//2 - 8 : len(array)//2 + 8]
    # print(np.max(array))
    count = np.unique(np.where(array == 1)[0])
    num = np.sum(array > 0.5)
    print(num, np.max(array), array.shape[0], array.shape[1], array.shape[2])
    print(patient, 'true' if len(count) >= 3 else 'false')
    if len(count) >= 3 and num >= 10000: 
        output_file1.write(f"{patient.split('.')[0]}\n")


# print("\n\n\n\n\n")

for patient in os.listdir(path2):
    patient_path = path2 + patient
    array = sitk.GetArrayFromImage(sitk.ReadImage(patient_path))
    tensor = torch.from_numpy(array).type(torch.int16)
    array = tensor.numpy()
    array = array[len(array)//2 - 8 : len(array)//2 + 8]
    # array = array / 255
    array[array >= 0.5] = 1
    array[array < 0.5] = 0
    # image = sitk.GetImageFromArray(array)
    # sitk.WriteImage(image, patient_path)
    # # print(np.max(array))
    count = np.unique(np.where(array==1)[0])
    num = np.sum(array > 0.5)
    print(num, np.max(array), array.shape[0], array.shape[1], array.shape[2])
    # print(count)
    print(np.max(array), array.shape[0], array.shape[1], array.shape[2])
    print(patient, 'true' if len(count) >= 3 else 'false')    
    if len(count) >= 3 and num >= 10000 :
        
        output_file2.write(f"{patient.split('.')[0]}\n")
        
# path3 = "/data3/wangchangmiao/jinhui/seg_private/preprocessed_image/1.nii.gz"
# array = sitk.GetArrayFromImage(sitk.ReadImage(path3))
# print(np.max(array), array.shape[0], array.shape[1], array.shape[2])

# path4 = "/data3/wangchangmiao/jinhui/DATA/multitask/data3d/preprocessed_image/49.nii"
# array = sitk.GetArrayFromImage(sitk.ReadImage(path4))
# print(np.max(array), array.shape[0], array.shape[1], array.shape[2])