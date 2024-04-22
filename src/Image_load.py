import requests
import rasterio
import numpy as np
import csv

City_list = ['Zurich', 'Geneva', 'Basel', 'Lausanne', 'Bern', 'Winterthur', 'Luzern', 'St_Gallen', 'Lugano', 'Biel']

rgb_array = np.array([[0.2989], [0.5870], [0.1140]])

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], rgb_array).squeeze()


with open('data/full_list.csv', 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(['Location', 'Coordinates', 'Array'])

for city in City_list:
    path = f'data/city_link/{city}.csv'

    with open(path, 'r') as link_list:
        reader = csv.reader(link_list)
        next(reader)  
        for link in reader:
            try:
                response = requests.get(link[0])
                if response.status_code == 200:
                    with open('swissimage.tif', 'wb') as f:
                        f.write(response.content)

                    with rasterio.open('swissimage.tif') as dataset:
                        img_array = dataset.read([1, 2, 3])
                        metadata = dataset.meta
                        corner = metadata['transform']
                        center_x, center_y = corner * (metadata['width']/2, metadata['height']/2)
                        coord = (center_x, center_y)

                        gray = rgb2gray(img_array)

                        gray_list = gray.round().astype(int).flatten().tolist()

                        with open('data/full_list.csv', 'a', newline='') as file:
                            csvwriter = csv.writer(file)
                            csvwriter.writerow([city, coord, gray_list])
            except Exception:
                print(f"Failed to process")
    print(city + ' done')