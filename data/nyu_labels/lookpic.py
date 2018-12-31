import cv2
from PIL import Image
import numpy as np

imgs = []

for i in range(10):
    imgs.append(np.array(Image.open(("../nyu_labels/" + str(i) + ".png"))))
a = np.max(imgs)
b = np.where(imgs == 255)
a = 1