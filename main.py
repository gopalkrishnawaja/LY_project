from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import streamlit as st
import pickle
#import spacy
import re

#nlp = spacy.load("en_core_web_sm")

#@st.cache
#def loadmodel():
#    model = load_model("M3P_512_spam.h5")
#   return model 
# model = loadmodel()

model = load_model("M3P_512_spam.h5")
with open('tokenizer1.pickle', 'rb') as f:
      tokenizer=pickle.load(f)

st.header("Spam Classifier")
st.subheader("Enter the message you want to analyze")
text_input = st.text_area( "Enter sentence",height=50)


if st.button("Analyze"):
    
    print("Result")
    pattern = r'((http[s]*)?(:\/\/)?(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    pattern2= r'[0-9]+'
    
    text_input= re.sub(pattern,"",text_input)
    st.write(text_input)
    text_input= re.sub(pattern2,"",text_input)
    st.write(text_input)
    text_input= text_input.replace("+","")
    st.write(text_input)
    #doc=nlp(text_input)
    #text= " ".join([token.lemma for token in doc  if not (token.is_stop or token.is_punct)])
    encoded = tokenizer.texts_to_sequences([text_input])
    st.write(encoded)
    padded = pad_sequences(encoded, maxlen=76, padding='post')
    st.write(padded)
    ypred = model.predict([padded,padded,padded])>0.5
    st.write(ypred)