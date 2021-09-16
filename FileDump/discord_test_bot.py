import praw
import tensorflow as tf
import pandas as pd
import numpy as np
import spacy
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    text = text.lower() 
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text) 
    text = text.replace('x', '')
    words = [word for word in text.split() if word.lower() not in sw_spacy]
    text = " ".join(words)
    return text

MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

word_index = tokenizer.word_index

model = tf.keras.models.load_model('depression_model.h5')

def isDepressing(text):
  lst = {'body':[text]}
  # Calling DataFrame constructor on list  
  dframe = pd.DataFrame(lst) 
  dframe = dframe['body'].apply(clean_text)
  dframe = dframe.str.replace('\d+', '')
  tokenizer.fit_on_texts(dframe.values)
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))
  X1 = tokenizer.texts_to_sequences(dframe.values)
  X1 = pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
  print('Shape of data tensor:', X1.shape)
  a = model.predict(X1)
  print(a)
  if(a[0][0] > a[0][1]):
    return 0
  return 1

reddit = praw.Reddit(
    client_id="cs0byWCuk46QyL4LTQWxVA",
    client_secret="-W5vwUVM7iu4A_f71ey4HawAKavutw",
    user_agent="<console:test:1.0>",
)

subreddit = reddit.subreddit("depression")

for post in subreddit.hot(limit=10):
    print('**********************')
    print(post.title)
    x = isDepressing(post.title)
    print(x)
    

