---
title: Telco Churn Model
publishDate: 2024-07-01
img: /assets/projects/telcochurnmodel/header.webp
img_alt: Telco Churn Model
description: This project focuses on developing a Churn Prediction Model to help businesses identify customers who are likely to leave their services in the near future
tags:
    - Data Cleaning
    - Machine Leaning
    - Docker
    - DevOps
pinned: true
github: https://github.com/IgnacioCorrecher/churn-model-SDG
---

# Churn Prediction Model

## Overview

This project focuses on developing a Churn Prediction Model to help businesses identify customers who are likely to leave their services in the near future. By accurately predicting churn, companies can implement targeted strategies for customer retention, such as segmentation, marketing campaigns, and loyalty programs.

The project follows a structured approach that includes data analysis, model development, and industrialization of the solution, ultimately leading to actionable insights for minimizing customer churn.

## Table of Contents

1.  [Context](#context)
2.  [Solution Architecture](#solution-architecture)
3.  [Exploratory Data Analysis (EDA) and Data Preparation](#exploratory-data-analysis-eda-and-data-preparation)
4.  [Machine Learning Models](#machine-learning-models)
5.  [Industrialization](#industrialization)
6.  [Results and Conclusions](#results-and-conclusions)

## Context

### Problem Statement

Understanding customer churn is crucial for any business. Identifying which customers are likely to leave allows companies to take preemptive actions to retain them. This project aims to present a comprehensive solution to predict churn using a provided dataset. The solution encompasses the entire data lifecycle, from raw data to predictive insights.

## Solution Architecture

The solution is structured into several key phases:

1. **Data Cleaning**: Initial processing to handle missing values, outliers, and high correlations between variables.
2. **Model Selection and Training**: Multiple machine learning models are trained and evaluated to select the best-performing model.
3. **Solution Industrialization**: The solution is made production-ready by integrating it into a pipeline for continuous monitoring and retraining.

## Exploratory Data Analysis (EDA) and Data Preparation

### Dataset Analysis

The dataset consists of 100 columns and 100,000 observations. A thorough analysis of the dataset was conducted to understand the underlying patterns and correlations, particularly focusing on the correlation with the churn variable.

### Data Cleaning

The data preparation phase involved handling missing values, outliers, and high correlations between features. These steps are crucial for improving the quality of the dataset and ensuring better model performance.

### Data Visualization

A custom web application was developed to visualize the distribution of variables used in the training process. This dashboard allows for a clean and interactive way to inspect the data, ensuring that the data quality is maintained throughout the process.

## Machine Learning Models

### Scaling and Encoding

-   **Scaler**: RobustScaler was used to normalize the data, as it is less sensitive to outliers compared to other methods.
-   **Encoder**: OneHotEncoder was applied to convert categorical variables into a numerical format, making them suitable for machine learning models.

### Model Pipeline

The dataset was split into training and testing sets while maintaining the proportions of the churn variable. A machine learning pipeline was created, incorporating the scaler, encoder, polynomial features, and feature selection steps.

### Model Training and Selection

Four different models were trained using cross-validation:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. K-Nearest Neighbors (KNN)

The model with the best performance, based on recall, was selected for final evaluation. The chosen model was a Random Forest, which achieved a recall score of 0.699 on the test set.

### Metric Selection

The recall metric was chosen to prioritize the identification of customers likely to churn, as false negatives (customers who churn but are not predicted to) can be particularly costly.

## Industrialization

### Technologies Used

To ensure the solution is ready for deployment, three main technologies were utilized:

1. **Docker**: For creating consistent and portable environments.
2. **Apache Airflow**: For orchestrating tasks and workflows through Directed Acyclic Graphs (DAGs).
3. **MLflow**: For managing and monitoring different iterations of the machine learning models, ensuring reproducibility and compatibility.

### Adaptation for Industrialization

The codebase was refactored to align with object-oriented programming principles. Two main classes were created: `DataCleaner` for preprocessing and `ModelTrainer` for training the model. These were integrated into a DAG that orchestrates the data cleaning, model training, and deployment tasks in sequence.

## Results and Conclusions

### Model Performance

The Random Forest model achieved a recall score of 0.699 on the test dataset, indicating a significant improvement over a random model that would achieve about 50% accuracy. The low number of false negatives demonstrates the model's effectiveness in identifying customers who are likely to churn.

### Interpretability and Feature Importance

The model's interpretability was enhanced by analyzing feature importance, which measures the impact of each variable on the model's predictions. This insight helps in understanding which factors contribute most to customer churn.

### Future Improvements

Potential improvements include moving from binary classification to a probabilistic model that estimates the likelihood of each customer churning. This would allow for a more nuanced analysis and enable cost-benefit decisions based on customer importance.

## Conclusion

The Churn Prediction Model provides a robust framework for identifying customers at risk of leaving. By leveraging machine learning, the solution offers actionable insights that can be used to enhance customer retention strategies, ultimately driving business success.
