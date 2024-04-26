import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v3'

    st.info(
        f"Distribution of images per set and label, performance metrics and brief explanations of the results."
    )

    st.write("### Distribution of images per set and label ")

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution_table.png")
    st.image(labels_distribution, caption='Distribution of images in each dataset in a table')

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution_bar.png")
    st.image(labels_distribution, caption='Distribution of images in each dataset visualized in a bar chart')

    labels_distribution = plt.imread(f"outputs/{version}/labels_pie_chart.png")
    st.image(labels_distribution, caption='Distribution of images total in a pie chart')

    st.warning(
        f"The images are divided into three subsets. \n"
        f"+ Train set (70%). \n"
        f"  + This is the dataset the model is trained on. It is the largest protion of the data, "
        f"enabling the model to better generalize the data and increase performance on unseen data. \n"
        f"+ Validation set (10%) \n"
        f"  + The validation dataset is used during the training phase to evaluate performance of the model. \n"
        f"+ The test set (20%) \n"
        f"  + After the model training, the model uses the test dataset to conduct evaluations on unseen data."
    )
    st.write("---")

    st.write("### Model Performance")

    model_clf = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_clf, caption='Confusion Matrix')  

    st.warning(
        f"**Confusion Matrix**\n\n"
        f"In binary classification, a confusion matrix is a table that provides a summary of "
        f"the model's performance by comparing predicted labels with actual labels. \n"
        f"The matrix has two rows and two columns, representing the two classes being predicted (healthy and mildew) and the two actual classes (true and false). \n\n"
        f"Explanation of the terms withing a confusion matrix: \n\n"
        f"+ True Positive (TP): \n"
        f"  + This represents the case where the model correctly predicts the positive (healthy) class. \n\n"
        f"+ True Negative (TN): \n"
        f"  + This represents the case where the model correctly predicts the negative (mildew) class. \n\n"
        f"+ False Positive (FP): \n"
        f"  + This occurs when the model incorrectly predicts the positive class when the actual class is negative. \n\n"
        f"+ False Negative (FN): \n"
        f"  + This occurs when the model incorrectly predicts the negative class when the actual class is positive. \n\n"
        f"A well trained model has high TP and TN rates while keeping the FP and FN rates low."
    )

    st.write("### Performance on test set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Accuracy', 'Loss']))

    st.warning(
        f"**General performance on test set** \n\n"
        f"The Accuracy value reflects the proportion of correctly classified images out of the total images in the dataset. \n"
        f"The Loss value measures the discrepancy between the true labels and predicted labels produced by the model during training."
    )

    model_clf = plt.imread(f"outputs/{version}/model_accuracy_training.png")
    st.image(model_clf, caption='Accuracy during training')

    st.warning(
        f"**Accuracy during training** \n\n"
        f"As displayed in the line graph above, the model initially achieves a high level of accuracy "
        f"on the training data and continues to improve very fast to a higher accuracy. The validation accuracy, "
        f"which measures the models performance on unseen data, drops for a short period of time below "
        f"the accuaracy line. That could indicate, that the model is overfitting or model instability."
    )

    model_clf = plt.imread(f"outputs/{version}/model_loss_training.png")
    st.image(model_clf, caption='Loss during training')

    st.warning(
        f"**Loss during training** \n\n"
        f"As displayed in the line graph above, the model initially achieves a slightly high level of loss "
        f"during the model training but continues to improve very fast to a lower loss level. "
        f"The model stabilizes at a low loss value, managing to predict the true labels."
    )

    st.write(
        f"For more information, please visit the "
        f"[**Project README.**](https://github.com/NixTS/ml-mildew-detection-in-cherry-leaves/blob/main/README.md)")