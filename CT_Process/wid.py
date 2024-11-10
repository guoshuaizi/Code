# import nibabel as nib
# import numpy as np
import SimpleITK as sitk
import os

def adjustMethod1(data_resampled,w_width,w_center):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = data_resampled.copy()
    data_adjusted[data_resampled < val_min] = val_min
    data_adjusted[data_resampled > val_max] = val_max

    return data_adjusted

def window(master_file,master_save_path,segmentation_file,segmentation_save_path):
    filenames = os.listdir(master_file)  # 读取nii文件夹
    slice_trans = []
    # center = 30  # 窗位
    # width = 300  # 窗宽
    # min = (2 * center - width) / 2.0 + 0.5
    # max = (2 * center + width) / 2.0 + 0.5
    #
    # dFactor = 255.0 / (max - min)



    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(master_file, f)
        ct = sitk.ReadImage(img_path, sitk.sitkFloat32)
        seg = sitk.ReadImage(os.path.join(segmentation_file,f), sitk.sitkFloat32)
        new_seg = sitk.GetArrayFromImage(seg)
        img_fdata = sitk.GetArrayFromImage(ct)
        #
        # img_fdata[img_fdata < min] = min
        # img_fdata[img_fdata > max] = max


        # 进行转置，因为需要按照原来的方向进行保存
        print('input_shape:', img_fdata.shape)
        data = img_fdata
        # (z, y, x) = data.shape
        # for i in range(z):
        #     for j in range(y):
        #         for k in range(x):
        #             value = data[i, j, k]
        #             if value <= min:
        #                 value = 0
        #             elif value < max:
        #                 value = (value - min) / width * 255
        #             elif value >= max:
        #                 value = 255
        #             else:
        #                 data[i, j, k] = value
        #             # 进行保存
        print("-----------------")

        data = adjustMethod1(data,300,30)
        print("output_shape:", data.shape)
        new_f = 'window_'+f

        img = sitk.GetImageFromArray(data)
        sitk.WriteImage(img, os.path.join(master_save_path,new_f))
        sitk.WriteImage(sitk.GetImageFromArray(new_seg), os.path.join(segmentation_save_path, new_f))

        print("+++++++++++++++++")




