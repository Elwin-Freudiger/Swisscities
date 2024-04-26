import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd

links = []

with open('data/all_links.csv', 'r') as file:
    links = file.read().splitlines()

print(links[5])

sub_list = np.random.randint(len(links)+1, size = 5000)

#formula to convert rgb image into a grayscale
rgb_array = np.array([[0.2989], [0.5870], [0.1140]])
def rgb2gray(rgb):
    return np.dot(rgb.T, rgb_array).squeeze().T


#open the csv and write the headers
with open('data/csv/All_images.csv', 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(['East', 'North', 'Array'])

#read every link and download it's content
for item in sub_list:
    try:
        response = requests.get(links[item])
        if response.status_code == 200:
            with open('swissimage.tif', 'wb') as f:
                f.write(response.content)
            #open image with rasterio
            with rasterio.open('swissimage.tif') as dataset:
                img_array = dataset.read([1, 2, 3]) #read rgb array
                metadata = dataset.meta #read metadata
                corner = metadata['transform']
                center_x, center_y = corner * (metadata['width']/2, metadata['height']/2)

                gray = rgb2gray(img_array) #transform into grayscale array
                gray_lower = cv.resize(gray, (100,100))
                gray_list = gray_lower.round().astype(int).flatten().tolist()
                #add line to csv
                with open('data/csv/All_images.csv', 'a', newline='') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow([center_x, center_y, gray_list])
    except Exception: #expection in case the link is broken
        print(f"Failed to process")
