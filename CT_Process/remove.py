import os
import cv2 as cv




segs_path = 'D:/bqgc/bai'   # 白色
imgs_path = 'D:/bqgc/output'   # 灰色

for filename in os.listdir(imgs_path):

    segs = cv.imread(segs_path + '/' + filename)[:, :, 0]

    imgs = cv.imread(imgs_path + '/' + filename)[:, :, 0]
    print(filename)
    pictue_size1 = segs.shape
    picture_height1 = pictue_size1[0]
    picture_width1 = pictue_size1[1]
    i1 = 0
    for a in range(picture_height1):
        for b in range(picture_width1):
            if segs[a, b].all() > 0:
                i1 = i1 + 1
    print('像素点为白的面积有:', i1)

    pictue_size2 = imgs.shape
    picture_height2 = pictue_size2[0]
    picture_width2 = pictue_size2[1]
    i2 = 0
    for c in range(picture_height2):
        for d in range(picture_width2):
            if imgs[c, d].all() > 0:
                i2 = i2 + 1
    print('像素点为灰的面积有:', i2)

    if i2 < i1 * 1.5:        # 灰色部分面积必须大于白色部分面积的1.5倍，否则删除
        print('删除')
        os.remove('D:/bqgc/output'+ '/' + filename)
