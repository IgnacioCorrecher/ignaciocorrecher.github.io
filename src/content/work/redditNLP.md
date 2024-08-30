---
title: Reddit NLP
publishDate: 2024-08-20
img: /assets/projects/redditNLP/header.webp
img_alt: Reddit Icon
description: We analyzed the gender pay gap in the US, visualized the data and developed several models to predict gender based on several factors.
tags:
    - Data Cleaning
    - Visualization
    - Machine Leaning
    - R
pinned: true
github: https://github.com/IgnacioCorrecher/reddit_NLP
---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Results](#results)
    - [Overall Sentiment Distribution](#overall-sentiment-distribution)
    - [Subreddit Sentiment Analysis](#subreddit-sentiment-analysis)
    - [Key Insights](#key-insights)

## Overview

`reddit_NLP` is a data science project focused on sentiment analysis of Reddit comments. This project processes a large dataset of 1 million comments from various subreddits and performs sentiment analysis to understand the overall mood and opinions expressed by users.

## Features

-   **Data Collection**: Utilizes Reddit's API to scrape 1 million comments from a diverse range of subreddits.
-   **Sentiment Analysis**: Implements various sentiment analysis techniques to classify comments as positive, negative, or neutral.
-   **Data Visualization**: Provides visual insights into the sentiment distribution across different subreddits and time periods.
-   **Customizable Parameters**: Allows users to adjust the number of comments or subreddits for data collection.

## Results

The sentiment analysis of 1 million Reddit comments yielded insightful results about user opinions across various subreddits. Key findings include:

### Overall Sentiment Distribution

<img src="/assets/projects/redditNLP/overall_sentiment.png" alt="Overall Sentiment Distribution" style="float: right; width: 50%; margin-left: 2rem;" />

The analysis revealed the following sentiment breakdown:

-   Negative: 67.84%
-   Positive: 32.16%

This distribution suggests that Reddit comments tend to lean more towards negative sentiment, with about two-thirds of the analyzed comments classified as negative.
<br><br><br><br><br><br><br><br><br>

### Subreddit Sentiment Analysis

<p align ="center"><img src="/assets/projects/redditNLP/subreddit_sentiment.png" alt="Layers of a CNN" style="width: 80%; margin-left: 2rem;" /></p>

The sentiment distribution varies across different subreddits, as visualized in the chart above. This analysis helps identify which subreddits tend to have more positive or negative discussions.

### Key Insights

1. The overall negative sentiment bias (67.84%) could be indicative of the critical nature of discussions on Reddit or reflect the topics that were most discussed during the data collection period.

2. Despite the overall negative trend, nearly a third of comments (32.16%) express positive sentiment, showing that there's still a significant amount of positive interaction on the platform.

3. The variation in sentiment across subreddits (as shown in the subreddit sentiment distribution chart) suggests that the topic or community significantly influences the tone of discussions.

4. This analysis provides a foundation for more in-depth studies, such as investigating reasons behind sentiment trends in specific subreddits or tracking sentiment changes over time.

These results offer valuable insights into public opinion and discourse trends on Reddit, which can be useful for various applications such as market research, social studies, or platform moderation strategies.

To explore the data further or to see examples of positive and negative comments, run the `visualize_results.py` script, which provides sample comments for each sentiment category.
