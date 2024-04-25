import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew of sweet and sour cherry is caused by Podosphaera clandestina, an obligate biotrophic fungus. "
        f"Powdery mildew on cherry leaves typically appears as a white to grayish powdery growth on the surface of infected plant tissues. "
        f"It can reduce photosynthesis, stunt growth, and decrease fruit quality, ultimately leading to the loss in affected plants. \n\n"
        f"A CNN model was trained to swiftly identify powdery mildew on cherry leaves, aiming to detect the infection. "
        f"The following visual criteria are used for identification and declaration:\n"
        f"+ White to grayish powdery patches on the leaf surfaces.\n"
        f"+ Light-green circular lesions may be visible. \n"
        f"+ The fungal growth can cover large areas of the leaf surface, giving it a fuzzy or velvety appearance. \n\n"
        f"For more information visit: [WSU Tree Fuit Site](https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/#:~:text=Powdery%20mildew%20of%20sweet%20and,1)")

    st.success(
        f"**The project's business requirements are:**\n\n"
        f"+ A study to visually differentiate healthy from an infected leaves.\n\n"
        f"+ A prediction whether a given leaf is infected by powdery mildew or not. \n\n"
        f"+ Download a prediction report of the examined leaves."
        )

    st.warning(
        f"**Project Dataset**\n\n"
        f"The dataset contains the follwing images: \n\n"
        f"+ 2104 - Healthy cherry leaves \n\n"
        f"+ 2104 - Cherry leaves infected with powdery mildew")

    st.write(
        f"For more information, please visit the "
        f"[**Project README.**](https://github.com/NixTS/ml-mildew-detection-in-cherry-leaves/blob/main/README.md)")