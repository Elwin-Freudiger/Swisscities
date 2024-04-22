import requests
import rasterio
from rasterio.plot import show
from PIL import Image
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tif_file = "src/swissimage-dop10_2023_2541-1160_2_2056.tif"

with rasterio.open(tif_file) as image:
    img_array = image.read([1,2,3])
    metadata = image.meta
    corner = metadata['transform']
image.close()

top_left_x, top_left_y = corner * (0, 0)
center_x, center_y = corner * (metadata['width']/2, metadata['height']/2)

print("Top left corner coordinates:", top_left_x, top_left_y)
print("Bottom right corner coordinates:", center_x, center_y)

rgb_array = np.array([[0.2989], [0.5870], [0.1140]])
def rgb2gray(rgb):
    return np.dot(rgb.T, rgb_array).squeeze().T

#gray = rgb2gray(img_array)
#show(gray, cmap = 'terrain')
