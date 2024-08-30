---
title: Telco Churn Model
publishDate: 2024-07-01
img: /assets/projects/telcochurnmodel/header.webp
img_alt: Telco Churn Model
description: We analyzed the gender pay gap in the US, visualized the data and developed several models to predict gender based on several factors.
tags:
    - Data Cleaning
    - Visualization
    - Machine Leaning
    - R
pinned: true
---

# Table of Contents

1. [Context and introduction](#context-and-introduction)
2. [Variable Selection](#variable-selection)
3. [Tuning algorithms](#tuning-algorithms)
4. [Conclusion](#conclusion)
5. [PDF](#pdf)

<a name="context-and-introduction"></a>

## Context and introduction

The dataset titled “The Gender Pay Gap in the General Social Survey” was obtained from [this source](https://vincentarelbundock.github.io/). It contains information on individuals surveyed between 1974 and 2018, including their gender, occupation, age, education level, and more. The primary objective is to study and predict the gender variable, representing the individual's gender.

<a name="variable-selection"></a>

## Variable Selection

After reducing the dataset to 5000 observations while maintaining the original gender proportion, an exploratory visual analysis was conducted. Variables were categorized into continuous, categorical, and dependent variables, with the output variable coded as Yes/No based on gender. Missing values in the salary variable (realrinc) were imputed using the median of the respective occupational group.

To identify significant variables, a stepwise AIC method was employed, followed by repeated stepwise selection to refine the list. The final important variables included work status, occupation codes, realrinc, age, and several marital and prestige score categories.

<a name="tuning-algorithms"></a>

## Tuning algorithms

#### Neural Networks

A grid search was performed to optimize the number of nodes and learning rate for the avNNet model, with the best results achieved using size = 5 and decay = 0.01. Further tuning with 250 iterations improved model performance.

#### Bagging and Random Forest

Random Forest models were tuned for the mtry parameter, with the best accuracy achieved using mtry = 9. The importance of variables like occ10, work status, realrinc, and age was highlighted.

#### Gradient Boosting

Optimal parameters for the GBM model were determined using a grid search, with the best results obtained using n.trees = 5000, shrinkage = 0.05, and n.minobsinnode = 20. Variable importance analysis showed that occ10, realrinc, work status, and age were the most significant.

#### Support Vector Machines (SVM)

Linear, polynomial, and radial basis function (RBF) SVMs were tuned for optimal parameters, with the polynomial SVM showing the best performance among the SVM models.
<img src="/assets/projects/genderpaygap/model-comparison.webp" alt="Model Comparison" style="float: right; width: 50%; margin: 2rem 2rem;" />

#### Model Comparison and Ensemble

A comparative analysis using cross-validation showed that the GBM model consistently outperformed other models in terms of accuracy and AUC. Ensembles of the top models were also evaluated, further confirming GBM as the best model.

<a name="conclusion"></a>

## Conclusion

The Gradient Boosting (GBM) model emerged as the best performer with an accuracy of approximately 0.88. This model provided the highest AUC and the lowest error rate, making it the preferred choice for predicting gender based on the available dataset.

<a name="pdf"></a>

## PDF

Here you can find the extended analysis, including the results of the cross-validation and the comparison of the models.<br><br><br>

<embed src="/assets/projects/genderpaygap/GenderPayGap.pdf" type="application/pdf" width="100%" height="600px" />
