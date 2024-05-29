import cv2
import numpy as np

from skimage.metrics import mean_squared_error as compare_mse
from skimage.color import deltaE_ciede2000

def fmse(img1, img2):
    mse_value = compare_mse(img1, img2)
    return mse_value


def fciede(img1, img2):
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    ciede_values = deltaE_ciede2000(img1_lab, img2_lab).mean()
    
    return ciede_values
