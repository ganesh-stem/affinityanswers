# import the essential libraries
import pandas as pd 
import numpy as np 
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import spacy
import string

# import the dataset and list of slur words
dataset = pd.read_csv("input_data.csv")
text_file = open("badwords.txt", "rt")

nlp = spacy.load('en_core_web_sm')
snowball_stemmer = SnowballStemmer(language='english')
all_stopwords = nlp.Defaults.stop_words

# open the file having a list of slur words and save into the list named toxic words
text_file = open("badwords.txt", "rt")
toxic_words = []
for line in text_file:
  toxic_words.append(line[:-1])

text_file.close()

# temperary dataset variable
temp = dataset.loc[:, ["tweet"]]

# clean the dataset using regular expressions
def  clean_text(text):
  text =  text.lower()
  text = re.sub('@[^\s]+','',text)
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"\r", "", text)
  text = re.sub(r"he's", "he is", text)
  text = re.sub(r"she's", "she is", text)
  text = re.sub(r"it's", "it is", text)
  text = re.sub(r"that's", "that is", text)
  text = re.sub(r"what's", "that is", text)
  text = re.sub(r"where's", "where is", text)
  text = re.sub(r"how's", "how is", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can't", "cannot", text)
  text = re.sub(r"n't", " not", text)
  text = re.sub(r"n'", "ng", text)
  text = re.sub(r"'bout", "about", text)
  text = re.sub(r"'til", "until", text)
  text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
  text = text.translate(str.maketrans('', '', string.punctuation)) 
  text = re.sub("(\\W)"," ",text) 
  text = re.sub('\S*\d\S*\s*','', text)
  text = re.sub(r"rt", "", text)
  return text

# stem the words
def stem_words(getDataset):
  sentences = []
  for index, row in getDataset.iterrows():
    sentence = ' '.join([snowball_stemmer.stem(word) for word in row["tweet"].split()])
    sentences.append(sentence)

  return pd.DataFrame({"tweet": sentences})

# remove the stop words
def remove_stop_words(getDataset):
  sentences = []
  for index, row in getDataset.iterrows():
      sentence = ' '.join([word for word in row["tweet"].split() if word not in all_stopwords])
      sentences.append(sentence)

  return pd.DataFrame({"tweet": sentences})

# censor the toxic words and calculate the degree of profanity
def censor_toxic_words(temp):
  sentences = []
  toxicity_degree = []
  for index, row in temp.iterrows():
    new_sentence = ""
    words_list = []
    toxicity_counter = 0
    
    for word in row["tweet"].split():
      if word in toxic_words:
          new_sentence += '***** '
          toxicity_counter += 1
      else:
          new_sentence += word + ' '
        
    words_list.append(new_sentence)
    toxicity_degree.append(toxicity_counter)
    sentence = ' '.join([word for word in words_list])
    sentences.append(sentence)
  return pd.DataFrame({"tweet": sentences, "profanity_degree": toxicity_degree})

# calling clean_text() to clean the dataset using regular expression
result = map(clean_text, temp["tweet"])
temp = pd.DataFrame({"tweet": list(result)})

# stemming each word of a sentence in a dataset
temp = stem_words(temp)

# removing stopwords
temp = remove_stop_words(temp)

# censor toxic words and calculate the profanity degree based on
# the number of toxic words found in a sentence.
temp = censor_toxic_words(temp)

# output the result
temp.to_csv("output_data.csv")