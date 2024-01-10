import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "wayland"

import pytesseract
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open("./apples.webp"))
# norm = np.zeros(img.shape)
#
# lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
# l, a, b = cv.split(lab)
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
# cl = clahe.apply(l)
#
# limg = cv.merge((cl, a, b))
#
#
# img = cv.cvtColor(limg, cv.COLOR_Lab2RGB)
# img = cv.cvtColor(img_orig, cv.COLOR_RGB2GRAY)

# img = cv.medianBlur(img, 7)
img = cv.bilateralFilter(img, 3, 75, 75)
# img = cv.GaussianBlur(img, (3, 3), 3.0)

# img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)

# img2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            # cv.THRESH_BINARY,11,2)
# img2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)

# img = cv.addWeighted(img2, 0.5, img_orig, 0.5, 2.0)

# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img)

plt.show()


print(pytesseract.image_to_data(img))
