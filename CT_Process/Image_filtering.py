import cv2
import  os
import numpy as np

def tresholding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def canny_demo(image):
    t = 80
    canny_output = cv2.Canny(image, t, t * 2)
    # cv.imshow("canny_output", canny_output)
    # cv.imwrite("canny_output.png", canny_output)
    return canny_output

def cv_resize(img,h,w):
    [nh,nw]=img.shape[:2]
    if (nh*nw) < (h*w) :
        dst = cv2.resize(img,(h,w),interpolation = cv2.INTER_CUBIC)
    else:
        dst = cv2.resize(img,(h,w),interpolation = cv2.INTER_AREA)

    return dst
def DeL(segs_path,imgs_path):

    for filename in os.listdir(segs_path):

        segs = cv2.imread(segs_path + '/' + filename)
        imgs = cv2.imread(imgs_path + '/' + filename)

        mask = tresholding(segs)

        binary = canny_demo(mask)
        k = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, k)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contours)==1):
            flag = 1
            [h, w] = segs.shape[:2]
            for x in range(h):
                if (mask[x, 0] == 255 or mask[x, w - 1] == 255):
                    flag = 0
                    break
            for y in range(w):
                if (mask[1, y] == 255 or mask[h - 1, y] == 255):
                    flag = 0
                    break
            if flag==1:
                cv2.imwrite(("train/Labels/" + filename), cv_resize(segs,256,256))
                cv2.imwrite(("train/Images/" + filename), cv_resize(imgs,256,256))





