import os
import scipy.misc as cv2
import numpy as np

dataset = 'style'
saveset = 'style_crop'
size = 512

for fn in os.listdir(dataset):
    print fn
    img = cv2.imread(dataset + '/' + fn)
    w,h,c = np.shape(img)
    print w,h

    if w >= h:
        ratio = float(h)/float(w)
        resize_factor = (int(size/ratio), size)
        img_resize = cv2.imresize(img, resize_factor)
    else:
        ratio = float(w)/float(h)
        resize_factor = (size, int(size/ratio))
        img_resize = cv2.imresize(img, resize_factor)
    
    w,h,c = np.shape(img_resize)
    crop_w = int((w-size) * 0.5)
    crop_h = int((h-size) * 0.5)
#    cv2.imsave(saveset + '/' + 'resize_' + fn, img_resize)

    print crop_h, crop_w
    img_crop = img_resize[crop_w:crop_w+size,crop_h:crop_h+size,:]
    cv2.imsave(saveset + '/' + fn, img_crop)
    
 
