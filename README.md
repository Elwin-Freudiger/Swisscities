# City classification 
## Classifying Swiss cities based on Aerial Imagery

## Overview
This repository contains the files and the dataset for a project aimed at using a Machine Learning model to classify Swiss cities based on aerial imagery obtained through the *Federal Office of Topography*. The project is part of an Advanced Data Analytics class at the University of Lausanne (UNIL) under the supervision of Professor Simon Scheidegger.

## Introduction 

This project extrates color and texture features from an image and uses the Keras Functional API to classify these observations by City. 

## Structure

- **src:** contains the code to download and extract features from images.
- **prediction:** contains the code to create the model and fit it
- **data:** contains the download links, the final CSV's and the files used for additional contextual data
- **report:** contains various images and files to help in the final report.

## Abstract

"This project aims to explore the possibility of classifying cities based on Aerial imagery. The dataset used is $1km^2$ tiles taken from the Swiss Topographical Office. The model is trained to classify cities that the image is located inside, using several image feature such as pixel distribution, contrast and several other factors. 
Focusing on several cities in Switzerland, this project trains a Deep-neural-network model to achieve this.
This study demonstrates the feasibility of such a model and can have a wide range of applications."

## Contributions
Contributions to this project are welcome. If you have ideas for improvements, additional features, or bug fixes, feel free to build upon it.



# Swisscities

Features to add:

Gray intensity
RGB distribution

HOG features or edges

mean pixel
std dev

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


https://www.kaggle.com/code/datascientistsohail/texture-features-extraction
co_matrix = skimage.feature.graycomatrix(image, [5], [0], levels=256, symmetric=True, normed=True)

### Calculate texture features from the co-occurrence matrix
contrast = skimage.feature.graycoprops(co_matrix, 'contrast')
correlation = skimage.feature.graycoprops(co_matrix, 'correlation')
energy = skimage.feature.graycoprops(co_matrix, 'energy')
homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')


https://dev.to/haratena9/image-processing-2-rgb-histogram-5dii

https://learnopencv.com/edge-detection-using-opencv/

https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

https://keras.io/guides/functional_api/

https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor


