import requests
import rasterio
import numpy as np
import csv
import cv2 as cv

file_path = 'data/csv/Ten_cities_lower.csv'

new_size = (100,100)

#The list of cities we will use
City_list = ['Zurich', 'Geneva', 'Basel', 'Lausanne', 'Bern', 'Winterthur', 'Luzern', 'St_Gallen', 'Lugano', 'Biel']

#formula to convert rgb image into a grayscale
rgb_array = np.array([[0.2989], [0.5870], [0.1140]])
def rgb2gray(rgb):
    return np.dot(rgb.T, rgb_array).squeeze().T

#open the csv and write the headers
with open(file_path, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(['Location', 'East', 'North', 'Array'])

#loop for every city in our list
for city in City_list:
    path = f'data/city_link/{city}.csv'
    #open the file with all of download links
    with open(path, 'r') as link_list:
        reader = csv.reader(link_list)
        next(reader)  
        #read every link and download it's content
        for link in reader:
            try:
                response = requests.get(link[0])
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
                        gray_list = gray.round().astype(int).flatten().tolist()
                        #add line to csv
                        with open(file_path, 'a', newline='') as file:
                            csvwriter = csv.writer(file)
                            csvwriter.writerow([city, center_x, center_y, gray_list])
            except Exception: #expection in case the link is broken
                print(f"Failed to process")
    print(f'{city} done')