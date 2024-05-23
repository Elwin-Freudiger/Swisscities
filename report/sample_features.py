import rasterio
import cv2 as cv
import numpy as np
import skimage
import matplotlib.pyplot as plt

with rasterio.open('report/sample2.tiff') as file:
        img_array = file.read([1, 2, 3]).transpose((1, 2, 0))  # Correct dimension order for OpenCV
        metadata = file.meta
        corner = metadata['transform']
        center_x, center_y = corner * (metadata['width'] // 2, metadata['height'] // 2)

        mean = np.mean(img_array)
        print(mean)
        stdev = np.std(img_array)
        print(stdev)

        gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
        gray_dist = cv.calcHist([gray], [0], None, [256], [0,256])                    
        red = cv.calcHist([img_array], [0], None, [256], [0, 256])
        green = cv.calcHist([img_array], [1], None, [256], [0, 256])
        blue = cv.calcHist([img_array], [2], None, [256], [0, 256])
        distribution = np.column_stack((gray_dist.flatten(), red.flatten(), green.flatten(), blue.flatten()))
        fig, axs = plt.subplots(2,2)
        axs[0,0].plot(gray_dist, color='gray')
        axs[0,1].plot(red, color = 'red')
        axs[1,0].plot(green, color = 'green')
        axs[1,1].plot(blue, color='blue')
        axs[0,0].set_title('Grayscale distribution')
        axs[0,1].set_title('Red distribution')
        axs[1,0].set_title('Green distribution')
        axs[1,1].set_title('Blue distribution')
        fig.tight_layout()
        plt.savefig('report/color_distribution.png')
        plt.show()
        

        co_matrix = skimage.feature.graycomatrix(gray, [5], [0], levels=256, normed=True)
        contrast = skimage.feature.graycoprops(co_matrix, 'contrast')[0, 0]
        print(contrast)
        correlation = skimage.feature.graycoprops(co_matrix, 'correlation')[0, 0]
        print(correlation)
        energy = skimage.feature.graycoprops(co_matrix, 'energy')[0, 0]
        print(energy)
        homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')[0, 0]
        print(homogeneity)

