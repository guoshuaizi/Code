import SimpleITK as sitk
import os
"""
resample
"""


def resampleVolume(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0, 0, 0]
    inputdir = [0, 0, 0]
    #outspacing = [0.81,0.97,0.81]
    # 读取文件的size和spacing信息

    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()
    print("size:",inputsize)
    print("space:",inputspacing)
    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


def resample(file,savepath):
    # 读文件
    # file = 'labels'  # 你的nii或者nii.gz文件路径
    # savepath = 'save'
    for filename in os.listdir(os.path.join(file)):
        niipath = os.path.join(os.path.join(file), filename)
        vol = sitk.Image(sitk.ReadImage(niipath))
        print(filename)
        #print(niipath)
        #print(vol)
    # 重采样
        newvol = resampleVolume([1, 1, 1], vol)

    # 写文件
        wriiter = sitk.ImageFileWriter()
        wriiter.SetFileName( os.path.join(savepath,'{}'.format(filename)))
        wriiter.Execute(newvol)
