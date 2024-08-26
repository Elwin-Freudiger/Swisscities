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
