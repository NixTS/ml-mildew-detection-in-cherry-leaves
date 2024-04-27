# **Powdery Mildew Detector**

![mock up images of powdery mildew detector website](./readme_images/mockup.jpg)
[**Live Website**](https://ml-powdery-mildew-detector-d12411e2b28f.herokuapp.com/)

Powdery Mildew Detector is a data science and machine learning project.  
The business goal of this project is to detect a biothrophic fungus infection called [Powdery Mildew](https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/#:~:text=Powdery%20mildew%20of%20sweet%20and,1) in cherry leaves by utilizing a convulutional neural network (CNN) trained on images sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

## **Table of Contents**

1. [Dataset Contents](#dataset-contents)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [The ML Model](#the-ml-model)
5. [Implementation of the Business Requirements](#implementation-of-the-business-requirements)
6. [Dashboard Design](#dashboard-design)
7. [Bugs and Errors](#bugs-and-errors)
8. [Deployment](#deployment)
9. [Technologies](#technologies)
10. [Credits](#credits)

## **Dataset Contents**

The Dataset [cherry_leaves on Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) provided by [Code Institute](https://codeinstitute.net/global) has been used in this project.

The dataset features 4208 photographs of cherry leaves and the images are already split 50/50 into healthy and powdery mildew infected cherry leaves. Each images shows a single leaf either infected or uninfected against a neutral background and consistent lightning.  

[Back to top ⇧](#table-of-contents)

## **Business Requirements**

The primary objective of this project is to develop a machine learning model for the accurate detection of powdery mildew in cherry leaves. The model is intended to offer the client a reliable and fast way to detect and diagnose infected leaves. The client can effectively trace them back to their respective trees and take necessary measures, such as applying fungicides, to prevent the spread of the fungus.

Summary:
+ The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
+ The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
+ The client needs a dashboard.
+ The client wants a minimum prediction accuracy of 97%.


[Back to top ⇧](#table-of-contents)

## **Hypothesis and Validation**

### **Hypothesis**  

> Infected leaves are visually different from healthy ones.

### **Validation**

To validate the hypothesis, an image study must be conducted. The machine learning model must be trained to recognize these differences and reliably predict an infected from a healthy leaf.

> Healthy cherry leaves are characterized by their rich, dark green color, which often appear vibrant and glossy.

> Powdery mildew on cherry leaves typically appears as a white to grayish powdery growth on the surface of infected plant tissues.

### **Understanding the Problem**

Feeding an image dataset into a neural network model requires the images to be prepared before the model training starts. It is important to normalize the images, normalizing refers to the process of adjusting the pixel values of an image so they fall within a specific range.

+ This is done to make the image data more consistent an suitable for processing by machine learning models. 
+ Normalizing helps mitigating the effects of variations in lightning conditions, contrast and color distribution across the images in the dataset.

To complete this task, we need to calculate the mean and standard deviation of the dataset.

The mathematical formula takes the following dimensions of the images:

+ **H** is the height of an image, in our case in pixel (px)
+ **W** is the widht of an image, in our case in pixel (px)
+ **C** is the number of channel, in our case 3 für RGB images
+ **B** is the batch size, in our case 20 images per batch

The visual output for the mean image looks like this:

![average image for healthy leaves](readme_images/avg_var_healthy.png)
We notice the healthy leaves have a clear green center and more defined edges.

![average image for infected leaves](readme_images/avg_var_mildew.png)
The infected leaves show more white stripes in the center and and appear less distinct.

![difference between mean images](readme_images/avg_diff.png)
The most obvious pattern where we could differentiate the average images from one another is the center part.

[Back to top ⇧](#table-of-contents)