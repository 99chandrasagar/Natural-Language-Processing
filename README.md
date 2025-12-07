📘 NLP Notebook Collection

This repository contains a structured collection of Jupyter notebooks covering essential Natural Language Processing (NLP) techniques. Each notebook focuses on a specific concept, moving from basic preprocessing to modern word embedding techniques and neural embedding models.

📂 Contents

1️⃣ NLTK Preprocessing

File: NLTK Preprocessing.ipynb
A foundational notebook demonstrating classical NLP preprocessing with the NLTK library.

Topics Covered

Text cleaning

Tokenization (word/sentence)

Stopword removal

Stemming (Porter, Lancaster, Snowball)

Lemmatization

POS tagging

N-grams and frequency distribution

2️⃣ spaCy NLP Pipeline

File: Spacy.ipynb
A modern approach to NLP using spaCy’s industrial-strength pipeline.

Topics Covered

Tokenization and linguistic features

POS tagging

Named Entity Recognition (NER)

Lemmatization

Dependency parsing

Text similarity

Working with spaCy models (en_core_web_sm)

3️⃣ Word Embedding Techniques

File: Word_Embedding.ipynb

This notebook introduces the theory and application of distributed word representations.

Topics Covered

Bag-of-Words (BoW) representation

One-hot encoding

TF-IDF explanation

Introduction to word embeddings

Limitations of traditional methods

Importance of dense vector representations

4️⃣ CBOW & Skip-Gram (Word2Vec Architecture)

File: cbow_Skipgram.ipynb

A practical exploration of neural embedding models using the Word2Vec framework.

Topics Covered

Understanding CBOW (predict target from context)

Understanding Skip-Gram (predict context from target)

Comparison of both architectures

Training shallow neural networks for embeddings

Hyperparameters (window size, negative sampling, vector dims)

5️⃣ Word2Vec Case Study

File: Word2vec_casestudy.ipynb

A detailed case study applying word embeddings to real-world text.

Topics Covered

Why TF-IDF fails to capture context

Training Word2Vec on custom data

Visualizing embeddings

Semantic similarity & nearest neighbors

Practical examples of analogy reasoning

Interpretation of embedding space

🧰 Technologies Used

Python 3.x

Jupyter Notebook

NLTK, spaCy

Gensim / Word2Vec

NumPy, Pandas, Matplotlib
