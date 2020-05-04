# ted_prediction
Ted Talk prediction using Kaggle data

Use NLP to predict quality of Ted talks. Data set is from Kaggle: https://www.kaggle.com/rounakbanik/ted-talks Contains two files; one is transcripts and urls, the other is talk meta data such as talk tag.

This script builds a sentiment score for each talk and then splits the talk scores in to three bins: A,B,C. Then, it uses talk tags and transcript word stems (filtered by correlation with score) to develop a dummy table for use with two classifiers: neural net and XGBoost.
