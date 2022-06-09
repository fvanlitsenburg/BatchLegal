#Install
#!python -m spacy download en_core_web_sm
#!pip install spacy-lookups-data
#!pip install bertopic

#Imports

import sys
import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import *

import spacy

from bertopic import BERTopic #BERTtopic-model: https://github.com/MaartenGr/BERTopic

from umap import UMAP
from typing import List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


#Functions

def loading_data(csv):
    """Takes a csv that includes columns with the EU-(Sub-)Directories (dir_1, dir_2, dir_3)
    and a column with the documents' text and transforms it into a DataFrame"""
    data = pd.read_csv(csv)
    return data


def cleaning(sentence):
    """Function that should be applyed to the pd.DataFrame-Column with the documents' texts to preprocces the text-data."""

    # Bulding a list that includes semantically unimportant words of high frequency
    ignore_list = ['shall', 'may', 'regulation', 'article', 'union', 'state', 'eu', 'official',  'member', 'commission', 'accordance', 'european',
               'column', 'costum', 'nomenclature', 'directive', 'council', 'journal', 'information', 'agency', 'mssg', 'etf', 'mdssg', 'ltd', 'annex',
              'reg', 'solas', 'resmsc', 'hsc', 'ed', 'res', 'incl', 'corr', 'msccirc', 'msc', 'ec', 'eec', 'no', 'en', 'code', 'ii', 'iii',
               'xi', 'goi', 'pepp', 'imo', 'mm', 'unka', 'prask', 'tsg']

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercasing
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## removing numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## removing punctuation
    tokenized_sentence = word_tokenize(sentence) ## tokenizing
    stop_words = set(stopwords.words('english')) ## defining stopwords
    tokenized_sentence_cleaned = [w for w in tokenized_sentence
                                  if not w in stop_words] ## remove stopwords

    tokenized_sentence_cleaned = [w for w in tokenized_sentence_cleaned if not w in ignore_list]

    sentences = ' '.join(word for word in tokenized_sentence_cleaned)

    #spacy
    nlp = spacy.load('en_core_web_sm', disable=["tok2vec", "tagger", "parser", "attribute_ruler"])
    nlp.remove_pipe("lemmatizer")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()
    doc = nlp(sentences)
    lemmatized = " ".join([token.lemma_ for token in doc])

    return lemmatized


def model_to_figure(data, dir, name):

    """This functions takes a sub-directory of the preprocessed data_set and extracts all the information
    that is needed to run the three BERTtopic-visualizations"""

    df_selec = data.loc[data[f'dir_{dir}'] == name]
    txt = df_selec.Content
    txt = txt.tolist()

    #Training model
    model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    topics, probs = model.fit_transform(txt)

    #Building Dict
    dictionary = {"Sub_dir Name:": name, "get_topic": model.get_topics(), "topic_freq": model.get_topic_freq(), "topic_sizes": model.topic_sizes}

    #Getting Embeddings for the visualize_topic_function
    freq_df = model.get_topic_freq()
    topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [model.topic_sizes[topic] for topic in topic_list]
    words = [" | ".join([word[0] for word in model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    if len(topics) > 1:
        embeddings_1 = model.c_tf_idf.toarray()[indices]
        embeddings = MinMaxScaler().fit_transform(embeddings_1)
        embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(embeddings)
    else:
        embeddings = []

    if len(topics) != 1:
        distance_matrix = 1 - cosine_similarity(embeddings_1)
    else:
        distance_matrix = []

    return dictionary, embeddings, distance_matrix


def get_viz_data(data, dir, directory):
    """This function takes the preprocessed data, the number of the EU-directory
    and a list of all the themes of this very EU-directory.
    It returns a DataFrame and two lists that the functions in bert_viz.py take as arguments to visualize the topics."""
    topics = []
    embeddings = []
    distances = []

    for name in directory:
        temp = name.split(' ')
        nam = temp[0].replace(',', '')
        embeds = nam+'_embeds'
        dist = nam+'_dist'

        print("* * *")
        print(nam, embeds, dist)

        nombre_1 = embeds
        nombre_2 = dist

        nam, embeds, dist = model_to_figure(data, dir, name)

        embed = {f'{nombre_1}': embeds}
        distance = {f'{nombre_2}': dist}

        topics.append(nam)
        embeddings.append(embed)
        distances.append(distance)

    return pd.DataFrame(topics), embeddings, distances
