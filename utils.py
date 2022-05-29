import numpy as np


def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1
    return img.reshape(shape)