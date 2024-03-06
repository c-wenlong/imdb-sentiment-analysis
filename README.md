# IMDb Sentiment Analysis

This project focuses on analyzing the sentiment of movie reviews from IMDb. Using natural language processing (NLP) techniques and machine learning models, this project aims to classify reviews as positive or negative based on their content.

## Overview

The sentiment analysis is performed on a dataset of IMDb movie reviews. The goal is to predict the sentiment of a review—whether it's positive or negative—using the text of the review. I am using 2 models in particular, the [twitter-roberta](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and [BERT-multilingual](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) models.

## Dataset

The [dataset](https://huggingface.co/docs/datasets/quickstart) used in this project consists of movie reviews from IMDb. Each review in the dataset is labeled as either positive or negative. For the purposes of this analysis, a subset of the dataset is used to train and test the sentiment analysis model.

## Dependencies

- python 3.X
- pandas
- datasets
- transformers
- torch
- scipy
- scikit-learn
- tqdm
- Matplotlib
- Seaborn

## Files in the Repository

- `movie_sentiment_analysis.ipynb`: The Jupyter notebook containing the sentiment analysis code, including data preprocessing, evaluation, and visualization.

## How to Run

To run this project, you will need Jupyter Notebook or Jupyter Lab installed on your machine. You can also use Google Colab if you prefer working in a cloud environment. Follow these steps:

1. Clone the repository or download the `movie_sentiment_analysis.ipynb` file.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open the `movie_sentiment_analysis.ipynb` notebook in Jupyter Notebook/Lab or Google Colab.
4. Run the cells in the notebook to perform the sentiment analysis.

## Methodology

The project follows these steps for sentiment analysis:

1. **Data Loading**: Load the movie reviews dataset.
2. **Data Preprocessing**: Clean and preprocess the text data for analysis, including tokenization and vectorization.
3. **Model Training**: Train a machine learning model on the preprocessed text data.
4. **Evaluation**: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
5. **Visualization**: Visualize the results using confusion matrices.

## Results

In general, both models performed very well in terms identifying negative comments, whereas positive comments are not as accurate. Although not significant, the BERT model did perform better than the twitter roberta model. I used the roberta model because it was the most downloaded model at the time of making this (48.3M downloads). That's pretty insane. However, the second model, with its rating system (out of 5), seemed more fitting for the specific task of analysing movie rating.
