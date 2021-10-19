# Reddit Post Classification

---
## Content
 - Problem Statement
 - Executive Summary
 - Summary of Analysis
 - Conclusion
 - Opportunity for Further Study

## Problem Statement
The goal of this project is to build a binary classification model to predict if a post from reddit belongs to the "War" or "Politics" subreddit. The model will be chosen if the Accuracy and F1 score are the highest. 

This project also aims to provide insights on the most discussed topic on each subreddit post by identifying the  words used against the score of the chosen model. 

## Data Dictionary
|Feature Name|Type|Description|
|----|----|----|
|subreddit|Integer|Binary Variable, War = 1, Politic = 0|
|message|object|combined title and selftext of 1 subreddit|
|message_lemma|object|lemmatized message|
|num_words|integer|number of words in each document|

## Executive Summary

### 01 - Data Collection
 - Libraries import
 - Requests post from Reddit - PushShift API
 - Save dataset as .csv
 
### 02 - Data Cleaning and EDA
 - Libraries import
 - Load data
 - Data Cleaning
    - Handling missing data
    - Lowercase all strings
    - Remove html tags and punctuation using regex
 - Feature Engineering 
    - Selecting features
    - Concating 2 subreddit post into 1 dataframe
    - Mapping target
- Explortary Data Analysis
    - Vizualizing most words used in wordcloud
    - Top 25 words for both subreddits
    - Top 25 words for each subreddits
- Save the cleaned data as .csv

### 03 - Modeling
- Libraries import
- Load data
- Defining X and Y variables
- Identifying the Baseline Score
- Split the data in train/ test
- Set up Pipeline
    - Count Vectorizer + Random Forest (cvec_rf)
    - Count Vectorizer + Logistic Regression (cvec_logreg)
- Set up parameters
- Set up dictionary to store the score
- Interpret, Evaluate and Select the best model
     - Confusion Matrix
- Rerun the best model with the best parameters
- Check feature importance

## Summary of Analysis

|Model|Train|Test|Accuracy|Misclassification|Sensitivity|Specificity|Precision|F1|Recall|ROC_AUC|
|----|----|----|----|----|----|----|----|----|----|----|
|cvec_rf|0.889697|0.829091|0.829091|0.170909|0.749091|0.909091|0.891775|0.81423|0.74909|0.94204|
|cvec_logreg|0.985455|0.898182|0.898182|0.101818|0.934545|0.861818|0.871186|0.90175|0.93455|0.97114|

 - [TP - Predicted positive and it's true](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) 
 - [TN - Predicted negative and it's true](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
 - [FP - Predicted positive and it's false (Type 1 Error)](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
 - [FN - Predicted negative and it's false (Type 2 Error)](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
 - [Accuracy - Overall, how often is the classifier correct](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [Misclassification Rate - Overall, how often is it wrong](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [Sensivitity/ Recall (TP Rate) - When it's actually yes, how often does it predict yes](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [FP Rate - When it's actually no, how often does it predict yes](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [Specificity (TN Rate) - When it's actually no, how often does it predict no](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [Precision - When it predicts yes, how often is it correct](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [F1 Score - Weighted average of the Recall and Precision](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
 - [ROC/ AUC Curve - ROC curve is used to summarize performance over all possible thresholds. it is the trade off between the true positive rate and false positive rate for a predictive model using the different probability thresholds](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
 - [ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

## Conclusion
From the baseline score of 50%, there was a lot of improvement when we used the models based from the table in Summary Analysis. However, different models have different scores when it comes to accuracy, F1, and ROC_AUC score. As mentioned in the problem statement, the chosen model will be with highest accuracy and F1 score. 
Based on the scoring table above, we can safely say that we choose Count Vectorizer + Logistic Regression (cvec_logreg) model as it has the highest score for accuracy and F1. However, the model seems to have an overfit which will be addressed as an improvement moving forward.


## Opportunity For Further Study
 - Develop a set of StopWords for the model which includes 'war' as a StopWords. 
 - Have more models so as to have more options and analyse the different scores which probably will have better score than only these two models. 
 - Set a wider range of hyperparameters tuning to reduce the overfitting issue. 
