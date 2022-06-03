#Imports
import streamlit as st
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import *


import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


st.write('hello, welcome to the BatchLegal interactive Dashboard!')

#Loading data from csv
data = pd.read_csv("../raw_data/test_data_scraped.csv")
st.write(data.columns)
df_content = data['content']

# list used to remove unrelevant terms
ignore_list = {'ec', 'no', 'european', 'commission', 'eu', 'union',
                   'article', 'directive', 'council', 'regulation', 'official',
                   'journal', 'article', 'information', 'agency', 'regulation',
                   'mssg', 'data', 'member', 'states', 'etf', 'mdssg', 'shall'
                  }

def cleaning(sentence):

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

    #tokenized_sentence_cleaned = [w for w in tokenized_sentence_cleaned if not w in ignore_list] COMMENTED IGNORE OUT!

    sentences = ' '.join(word for word in tokenized_sentence_cleaned)

    #spacy
    nlp = spacy.load('en_core_web_sm', disable=["parser"])

    doc = nlp(sentences)
    lemmatized = " ".join([token.lemma_ for token in doc])

    return lemmatized

# Applying Davy's Function

clean_txt = df_content.apply(cleaning)

# vectorization

vectorizer_n_gram = TfidfVectorizer(ngram_range = (1,1)) # "One"-GRAMS
cleaned_vectorizer_n_gram = vectorizer_n_gram.fit_transform(clean_txt)

# function for integration to .py file later
# def vectorizer(clean_txt):
#     vectorizer_n_gram = TfidfVectorizer(ngram_range = (1,1)) # BI-GRAMS
#     cleaned_vectorizer_n_gram = vectorizer_n_gram.fit_transform(clean_txt)
#     return cleaned_vectorizer_n_gram

# df = pd.DataFrame(cleaned_vectorizer_n_gram.toarray(), columns=vectorizer_n_gram.get_feature_names_out())

#Modelling

# Instantiating the LDA
n_components = 3
lda_model = LatentDirichletAllocation(n_components=n_components, max_iter = 100)

# Fitting the LDA on the vectorized documents
lda_model.fit(cleaned_vectorizer_n_gram)

# Getting topics
topics = lda_model.transform(cleaned_vectorizer_n_gram)

#Topic model function from ML-10-lecture
def print_topics(model, vectorizer, top_words):
    for idx, topic in enumerate(model.components_):
        st.write("-"*20)
        st.write("Topic %d:" % (idx))
        st.write([(vectorizer.get_feature_names_out()[i], round(topic[i],2))
                        for i in topic.argsort()[:-top_words - 1:-1]])


#Printing topics

st.write(print_topics(lda_model, vectorizer_n_gram, top_words = 8))

import numpy as np
# adding random dates to clean_txt
def random_dates(start, end, n=10):

    start_u = start.value//10**9
    end_u = end.value//10**9

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

start = pd.to_datetime('2015-01-01')
end = pd.to_datetime('2018-01-01')

viz_text = pd.DataFrame(clean_txt)
viz_text['date'] = random_dates(start, end,len(clean_txt))
