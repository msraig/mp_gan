import numpy as np

def toHDR_float(img):
    img = img
    img_out = img ** (2.2)
    return img_out.astype(np.float32)

def toLDR_float(img, scale = 1.0, clamp = True):
    img_out = scale * img ** (1.0 / 2.2)
    if(clamp):
        img_out = np.minimum(1.0, img_out)
    return img_out.astype(np.float32)
