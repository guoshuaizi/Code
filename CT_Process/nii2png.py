import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像


def nii_to_image(niifile,savepath,mode):
    filenames = os.listdir(niifile)  # 读取nii文件夹
    slice_trans = []

    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(niifile, f)
        img = nib.load(img_path)  # 读取nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii.gz', '')  # 去掉nii的后缀名

        # 开始转换为图像
        (x, y, z) = img.shape
        if mode == "axial":
           for i in range(x):
               silce = img_fdata[ i,:, :]
               imageio.imwrite(os.path.join(savepath, 'axial_{}_{}.png'.format(fname,i)), silce)

        if mode == "coronal":
           for i in range(y):
               silce = img_fdata[ :,i, :]
               imageio.imwrite(os.path.join(savepath, 'coronal_{}_{}.png'.format(fname,i)), silce)

        if mode == "sagittal":
           for i in range(z):
               silce = img_fdata[ :,:,i]
               imageio.imwrite(os.path.join(savepath, 'sagittal_{}_{}.png'.format(fname,i)), silce)



if __name__ == '__main__':
    filepath = 'E:\CTpython\sources_单肾\sources_单肾\case_master'
    savepath = 'Images'
    mode = ["axial", "coronal", "sagittal"]
    nii_to_image('data1230/ct/seg','data1230/2d-ct/seg',mode[0])
    nii_to_image('data1230/ct/seg', 'data1230/2d-ct/seg', mode[1])
    nii_to_image('data1230/ct/seg', 'data1230/2d-ct/seg', mode[2])

    nii_to_image('data1230/ct/master','data1230/2d-ct/master',mode[0])
    nii_to_image('data1230/ct/master','data1230/2d-ct/master', mode[1])
    nii_to_image('data1230/ct/master','data1230/2d-ct/master', mode[2])

    # nii_to_image('master', 'results', mode[0])