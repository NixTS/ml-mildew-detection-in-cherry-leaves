import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_leaves_visualizer_body():
    st.write("### Cherry Leaves Visualizer")
    st.info(
        f"A study that visually differentiates cherry leaves affected by powdery mildew from a healthy one.")

    st.success(
        f"Cherry leaves infected by the powdery mildew fungus have clear marks. " 
        f"The first symptom is a light-green, circular lesion on the leaf surface," 
        f"then a white to greyish cotton-like growth develops in the infected area.\n\n" 
        f"The images were photographed on a plain background and are mostly centered, "
        f"which helps in normalizing the images for further process "
        f"and training a convolutional neural network (CNN). \n\n"
        f"When working with images, it is important to prepare the images before using " 
        f"them to train a CNN. One way to do this is by 'normalizing' the images, which means "
        f"making sure they have a consistent features like color, brightness, form and more."
    )
    
    version = 'v3'
    if st.checkbox("Difference between average and variability image"):
      
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")
        avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_mildew.png")

        st.warning(
            f"We notice the healthy leaves have a clear green center and more defined edges." 
            )

        st.image(avg_healthy, caption='Healthy leaves - Average and Variability')

        st.warning(
            f"The infected leaves show more white stripes in the center and "
            f" and appear less distinct."
            )

        st.image(avg_powdery_mildew, caption='Powdery mildew infected leaves - Average and Variability')
        st.write("---")

    if st.checkbox("Differences between average infected and average healthy leaves"):
            diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

            st.warning(
                f"The most obvious pattern where we could differentiate the average images "
                f" from one another is the center part.")
            st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"): 
        st.write("To refresh the montage, click on the 'Create Montage' button")
        my_data_dir = 'input/cherry-leaves'
        labels = os.listdir(my_data_dir+ '/validation')
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)
        if st.button("Create Montage"):      
          image_montage(dir_path= my_data_dir + '/validation',
                        label_to_display=label_to_display,
                        nrows=8, ncols=3, figsize=(10,25))
        st.write("---")

    st.write(
    f"For more information, please visit the "
    f"[**Project README.**](https://github.com/NixTS/ml-mildew-detection-in-cherry-leaves/blob/main/README.md)")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  sns.set_style("white")
  labels = os.listdir(dir_path)

  if label_to_display in labels:

    images_list = os.listdir(dir_path+'/'+ label_to_display)
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return

    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)

  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")