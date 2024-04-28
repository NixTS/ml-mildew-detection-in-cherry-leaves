import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypotesis and Validation")

    st.info(
        f"Infected leaves are visually different from healthy ones."
    )

    st.warning(
        f"Healthy cherry leaves are characterized by their rich, dark green color, "
        f"which often appear vibrant and glossy. \n\n"
        f"Powdery mildew on cherry leaves typically appears as a white to grayish powdery "
        f"growth on the surface of infected plant tissues."
    )

    st.success(
        f"Through intensive training and fine tuning the model was able to detect such differences, "
        f"with an accuracy of more than 99%. \n"
        f"Testing the model with unfamiliar data confirmes this very favourable outcome. "
        f"This validates, that the model is able to generalize and predict reliably."
    )

    st.write(
        f"For more information, please visit the "
        f"[**Project README.**](https://github.com/NixTS/ml-mildew-detection-in-cherry-leaves/blob/main/README.md)")