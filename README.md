<img src="https://i.postimg.cc/K8rCzrRx/text-banner-removebg-preview.png" alt="ww" border="0"></a>

# NLP & Text Analytics
---

## Goodreads Book Reviews Classification

This repository contains the source code of the final project for the course " Text Analytics" at the University of Pisa.

The work done in this repository aims to evaluate reviews from web users using Artificial Intelligence (AI) and natural language processing (NLP) techniques with the ultimate goal of developing classification models useful for prediction tasks. 

## Phase 1: Retrieving Data
The dataset was retrieved through a challenge proposed on [**Kaggle**](https://www.kaggle.com/competitions/goodreads-books-reviews-290312/).

The latter was aimed towards the prediction of user ratings, instead it was decided to further develop the challenge into a genre prediction task with the objective of evaluating a book genre just from its reviews. 
Genres have been extracted from the original source inspired by the challenge: [**Goodreads.com**](https://www.goodreads.com/)

FInally, the datasets consist of 10 categorical and numerical features, but only `review_text` was used for the scope of the project, by also considering the `genre` target variable.

## Phase 2: Data Exploration and Preprocessing
After making sure of the goodness of the data, a pre-processing phase takes place, and different techniques have been applied to the reviews.

NLTK library was the one mostly adopted to fix texts: indeed `nltk.word_tokenize`, `nltk.corpus.stopwords`, `nltk.stem.wordnet.WordNetLemmatizer`, `nltk.pos_tag` and more have been used to make the reviews operable with classification models.

Each pre-processed version of the reviews has been been vectorized by the following: 

*  `Tokenizer` (provided by Keras, in general used for NNs)
*  `CountVectorizer`
*  `Tf-idf`
*  `Top2Vec` creating a new filtered column based on words similar to the different genre classes

## Phase 3: ML modelling

In this last phase several classifiers were tested and different performances have been identified based on the embeddings and strategies. 
In particular, the following classifiers were exploited for the classification task: LSTM, SVM, Random Forests, Naive Bayes.

Finally a state of the art Transformers (`BERT`) has been used to compare performance between classic ML models and text-ad hoc pretrained models.

---

## Tools

The main text tools adopted in this project are:

*  [**NLTK**]( https://www.nltk.org/) a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing etc.
*  [**Gensim**](https://radimrehurek.com/gensim/) a Python library for topic modelling, document indexing and similarity retrieval with large corpora
*  [**Spacy**](https://spacy.io/) for NER text representation
*  [**PyTorch**](https://pytorch.org/) to deal with Tensors
*  [**Transformers**](https://huggingface.co/docs/transformers/index) for pre-trained models.
*  [**Scikit-Learn**](https://scikit-learn.org/stable/) provides machine learning models and utility functions.


