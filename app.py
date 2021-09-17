import time
import flask
from flask import Flask, render_template, request, redirect, url_for
import re
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from spellchecker import SpellChecker
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
import pandas as pd

model = keras.models.load_model('Models/FinalModel.h5')

df = pd.read_csv('Datasets/MainData.csv')
df = df.dropna()

# load the pipeline object
pipeline = model

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cleanedText'].values)

def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
    words = [word for word in text.split() if word.lower() not in sw_spacy]
    text = " ".join(words)
    return text

def fuzzyLogic(text):
  spell = SpellChecker()
  text = ' '.join([spell.correction(word) for word in text.split()])
  return text

def get_ans(text_query):
    try:
        text = fuzzyLogic(text_query)
        lst = {'body':[text]}
        # Calling DataFrame constructor on list  
        dframe = pd.DataFrame(lst) 
        dframe = dframe['body'].apply(clean_text)
        dframe = dframe.str.replace('\d+', '')
        X1 = tokenizer.texts_to_sequences(dframe.values)
        X1 = pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', X1.shape)
        a = model.predict(X1)
        print(a)
        if(a[0][0] > a[0][1]):
            return 'The entered text shows no signs of depression!' #pd.Dataframe.from_dict({'prediction':0})
        return 'The entered text shows signs of depression.' #pd.Dataframe.from_dict({'prediction':1})
    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)

# function to get results for a particular text query
def requestResults(name):
    # get the tweets text
    ans = get_ans(name)
    return ans


# start flask
app = Flask('TrialAPI', template_folder='templates/')

# render default webpage
@app.route('/')
def home():
    return render_template('home.html')

# when the post method detect, then redirect to success function
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))

# get the data for the requested query
@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "


app.run()


