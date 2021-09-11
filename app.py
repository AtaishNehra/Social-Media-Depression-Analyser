import tkinter
from tkinter import *

window = tkinter.Tk()
width = 500
height = 550
window.minsize(width, height)
window.title("Depression Analyzer")
dep_analyzer = Label(window, text='Depression Analyzer',bg='black',font=('Arial',50),fg='white').pack()

text_fame = Frame(window,height = 100, bg='black')

# DO NOT TOUCH
dummy_frame1 = Frame(window,height=50,bg='black')
dummy_frame1.pack()

l1 = Label(text_fame, text="Enter post",bg='black',height=2, width=10,font=('Arial',40))
l1.pack()
e1 = Text(text_fame,bg='black',width=50)
e1.pack()

post = 2
def getPost():
    for widgets in dummy_frame.winfo_children():
      if(widgets.winfo_class() == 'Label'):
        widgets.destroy()
    global post
    post = predict(e1.get("1.0",'end-1c'))
    text = "Prediction: "
    if(post == 1):
        text += 'Depressed.'
    else:
        text += 'Not Depressed!'
    d1 = Label(dummy_frame,text=text,bg='black',font=('Arial',20))
    d1.pack(side='bottom')

def predict(text):
    from tensorflow import keras
    model = keras.models.load_model('Models/FinalModel.h5')
    import pandas as pd
    df = pd.read_csv('Datasets/MainData.csv')
    df = df.dropna()
    import re
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from spellchecker import SpellChecker
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    import spacy
    #loading the english language small model of spacy
    en = spacy.load('en_core_web_sm')
    sw_spacy = en.Defaults.stop_words

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

    def isDepressing(text):
        text = fuzzyLogic(text)
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
            return 0
        return 1
    
    return isDepressing(text)


# DO NOT TOUCH
dummy_frame = Frame(window,height=175,bg='black')
dummy_frame.pack(side='bottom')

button_frame = Frame(window,height=20, bg='black')
predict_button = Button(button_frame,text='Submit',fg='black',bg='black',command=getPost)
predict_button.pack(side='top')
button_frame.pack(side='bottom')
text_fame.pack()


window.configure(bg='black')
window.mainloop()