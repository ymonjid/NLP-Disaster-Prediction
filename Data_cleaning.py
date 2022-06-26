#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 19:04:38 2022

@author: ymonjid
"""
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import re
import string

# 1) Lemmatization + removing stopwords of the tweet text

lemmatizer = WordNetLemmatizer()
 
# Define function to lemmatize each word with its POS tag
 
# POS_TAGGER_FUNCTION : TYPE 1
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
 
def lemmatize(text_array):
    lemmatized_array = []
    for word in text_array.split(' '):    
        lemmatized_sentence = []
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(word))
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:       
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, wordnet.VERB))
        lemmatized_sentence = " ".join(lemmatized_sentence)
        lemmatized_array.append(lemmatized_sentence)  
    return ' '.join(lemmatized_array)


def clean_text(text):
    """ Make text lowercase, remove punctuation and words with alphanumeric characters"""
    text = text.lower()
    #text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w', '', text)
    text = re.sub('[ûò]', '', text)
    text = re.sub('[ò]', '', text)
    text = re.sub('[ï]', '', text)
    text = re.sub('a+[0-9]', '', text)
    text = re.sub('\w+[0-9]', '', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub('ó', '', text)
    text = re.sub('åê', '', text)
    text = re.sub('http' or 'https', '', text)
    
    return text

from geotext import GeoText
def get_places(text_arr):
    cities = [GeoText(x.capitalize()).cities for x in text_arr]
    cities = [','.join(x) for x in cities if x!=[]]
    countries = [GeoText(x.capitalize()).countries for x in text_arr]
    countries = [','.join(x) for x in countries if x!=[]]
    return ','.join(cities), ','.join(countries)

def data_cleaning(df):
    # 1) Cleaning the text column
    df['text'] = df['text'].apply(lambda x: lemmatize(x))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df['text'] = df['text'].apply(lambda x: clean_text(x))
    
    # 2) Cleaning the keyword column
    # a) Convert %20 to ' ' in the keyword
    df['keyword'] = df['keyword'].apply(lambda x: str(x).replace('%20', '-'))
    # b) Lemmatize keyword features
    df['keyword'] = df['keyword'].apply(lambda x: lemmatize(x))
    unique_keyword = df['keyword'].unique()
    # c) Replace NaN values in keyword by the common words between the list
    # of unique keywords in the df_train and the corresponding texts
    df.loc[df['keyword'] =='nan','keyword'] = df[df['keyword'] == 'nan']['text'].apply(lambda x:
           list(set(unique_keyword).intersection(set(x.split()))))
    # d) Clean the keyword column
    df['keyword'] = df['keyword'].apply(lambda x: ','.join(x))
    df['keyword'] = df['keyword'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', str(x)))
    
    # 3) Cleaning the location column
    df['location'] = df.apply(lambda x: ' '.join(get_places(x['text'].split(' '))) if x['location']!='NaN' else x['location'], axis = 1)
    
    return df
    
    
    
# 2) Remove the '%20' from the keywords and replace it by ' '
# 3) Dealing with the NaN values in the keyword and location columns
# 4) Detect keywords and locations in the text
    