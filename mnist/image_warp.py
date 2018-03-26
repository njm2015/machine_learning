import cv2
import numpy as np
import math

#####################
# Vertical wave
def vertical_wave(img, dim1, dim2):
    img = img.values.reshape((dim1,dim2))
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
            offset_y = 0
            if j+offset_x < rows:
                img_output[i,j] = img[i,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0

    return img_output.flatten()

#####################
# Horizontal wave
def horizontal_wave(img, dim1, dim2):
    img = img.values.reshape((dim1,dim2))
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = 0
            offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
            if i+offset_y < rows:
                img_output[i,j] = img[(i+offset_y)%rows,j]
            else:
                img_output[i,j] = 0

    return img_output.flatten()

#####################
# Both horizontal and vertical 
def horiz_vert_wave(img, dim1, dim2):
    img = img.values.reshape((dim1, dim2))
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
            offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
            if i+offset_y < rows and j+offset_x < cols:
                img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0

    return img_output.flatten()

#####################
# Concave effect
def concave(img, dim1, dim2):
    img = img.values.reshape((dim1,dim2))
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2*cols)))
            offset_y = 0
            if j+offset_x < cols:
                img_output[i,j] = img[i,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0

    return img_output