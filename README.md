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


