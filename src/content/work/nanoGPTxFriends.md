---
title: nanoGPT x Friends
publishDate: 2023-12-01
img: /assets/projects/nanoGPTxFriends/header.webp
img_alt: Friends Logo
description: |
    This project showcases a unique implementation of the Transformer architecture, inspired by the "Attention is All You Need" paper and Andrej Karpathy's tutorial series.
tags:
    - Data Cleaning
    - Visualization
    - Machine Leaning
    - R
pinned: true
github: https://github.com/IgnacioCorrecher/nanoGPTxFRIENDS
---

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Model Architecture](#model-architecture)
4.  [nanoGPT Model Explanation](#nanogpt-model-explanation)
5.  [Results](#results)
6.  [Acknowledgements](#acknowledgements)

## Introduction

This project is an exploration of the Transformer model, specifically focusing on its application in natural language generation. By training a nanoGPT model on the Friends TV Show Script dataset, we aim to create new, realistic episodes that capture the essence of the original show.

## Dataset

The Friends TV Show Script dataset, sourced from Kaggle, contains scripts from all 10 seasons of the show. This dataset is essential for training the model to generate dialogues that reflect the characters' personalities and the show's humor.

## Model Architecture

The nanoGPT model mirrors the original Transformer architecture, consisting of multiple layers of encoder-decoder structures. Key features include:

-   Multi-Head Self-Attention Mechanism
-   Positional Encoding
-   Feed-Forward Neural Networks
-   Layer Normalization and Dropout

## nanoGPT Model Explanation

This script implements a simplified version of the GPT (Generative Pre-trained Transformer) model using PyTorch. It is designed to generate text based on the "Friends" TV show transcripts. Below is a detailed explanation of each part of the script.

### Hyperparameters

The script starts by defining several key hyperparameters:

-   **Context Length**: This parameter defines the number of tokens the model considers simultaneously during training, which is also known as the block size.
-   **Batch Size**: The number of independent sequences processed in parallel during each iteration of training.
-   **Max Iterations**: The total number of training iterations.
-   **Evaluation Interval**: The frequency at which the model is evaluated on the validation dataset.
-   **Learning Rate**: The rate at which the model's weights are updated during training.
-   **Device**: The script checks whether a GPU is available and uses it if possible; otherwise, it defaults to the CPU.
-   **Embedding Dimension**: The size of the embedding vector that represents each token.
-   **Number of Attention Heads**: This defines how many separate attention mechanisms are used in the multi-head attention block.
-   **Number of Layers**: The total number of layers in the Transformer model.
-   **Dropout Rate**: A technique used to prevent overfitting by randomly setting a fraction of the input units to zero during training.

### Data Preparation

The script reads the "Friends" TV show transcript from a text file and processes it to create a vocabulary of unique characters. This vocabulary is then used to convert the text into a sequence of numerical indices, where each index corresponds to a character in the vocabulary.

Two dictionaries are created: one that maps characters to indices and another that maps indices back to characters. These dictionaries are used to encode strings into numerical sequences and decode numerical sequences back into strings.

### Train-Test Split

The entire dataset is divided into training and testing sets, with 95% of the data used for training and the remaining 5% for testing. This split ensures that the model is trained on a large portion of the data while still having a separate set for evaluation.

### Data Batching

To efficiently train the model, the script includes a function that generates batches of data. Each batch consists of sequences of tokens (input) and the corresponding sequences shifted by one position (target). The model learns to predict the next token in the sequence.

### Loss Estimation

The script includes a method for evaluating the model's performance. During evaluation, the model's predictions are compared against the actual sequences, and the average loss is calculated. This loss estimation helps monitor the model's performance on both the training and validation datasets.

### Model Architecture

The core of the script is the implementation of the nanoGPT model, which includes several key components:

-   **Multi-Head Self-Attention**: This mechanism allows the model to focus on different parts of the input sequence when making predictions. The multi-head setup means the model can attend to multiple positions in the sequence simultaneously, which improves its ability to capture complex relationships in the data.
-   **Feed-Forward Neural Networks**: After the attention mechanism processes the input, the data is passed through a series of fully connected layers. These layers help the model transform the input into a more useful representation for predicting the next token.
-   **Positional Embeddings**: Since Transformers do not inherently understand the order of tokens in a sequence, positional embeddings are added to the input embeddings to provide the model with information about the position of each token.
-   **Layer Normalization and Dropout**: These techniques are used to stabilize and regularize the training process, making the model more robust and reducing the likelihood of overfitting.

### Training Loop

The model is trained over a specified number of iterations. During each iteration, the script:

1. **Evaluates** the model at set intervals to monitor progress.
2. **Generates Batches** of training data.
3. **Computes Loss**: The difference between the model's predictions and the actual sequences is calculated.
4. **Backpropagation**: The model's weights are updated to minimize the loss using the AdamW optimizer.
5. **Generates Text**: After training, the model can generate new sequences of text by predicting one token at a time, starting from an initial context.

### Text Generation

Finally, the script includes a method for generating text using the trained model. Given an initial context (e.g., the first token or a few tokens), the model predicts the next token, appends it to the context, and repeats the process. This iterative generation continues until the desired length of text is achieved. The generated text is then saved to a file.

Overall, this script provides a practical implementation of a small-scale GPT model capable of learning from and generating text similar to the "Friends" TV show transcripts.

## Results

Here you can find an example of the generated output by the model:

<embed src="/assets/projects/nanoGPTxFriends/output.txt" type="application/pdf" width="100%" height="300px" />

## Acknowledgements

This project was inspired by:

-   [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.
-   [Andrej Karpathy's YouTube series](https://www.youtube.com/channel/UC6e2mP01ZLH_kbAyeazCNdg)
-   [Friends TV Show Script dataset](https://www.kaggle.com/datasets/rezaghari/friends-series-dataset)
