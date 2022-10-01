
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import pickle
import re
import string
import matplotlib.pyplot as plt
import time
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('dependency_treebank')
nltk.download('snowball_data')
nltk.download('rslp')
nltk.download('porter_test')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('pros_cons')
nltk.download('hmm_treebank_pos_tagger')
nltk.download('vader_lexicon')
nltk.download('treebank')
nltk.download('reuters')
nltk.download('universal_tagset')

@st.cache(allow_output_mutation=True)

def load(vectorizer_path, model_path):

    # Load the vectorizer.
    file = open(vectorizer_path, 'rb')
    vectorizer = pickle.load(file)
    file.close()
    
    # Load the Logistic Regression Model.
    file = open(model_path, 'rb')
    LRmodel = pickle.load(file)
    file.close()    
    return vectorizer, LRmodel

def predict(vectorizer, model, texts, cols):  
    text = texts.split(";")    
    finaldata = []
    
    textdata = vectorizer.transform((lemmatize_process((preprocess(text)))))
    sentiment = model.predict(textdata)
    sentimentp = model.predict_proba(textdata)
    
    for index,sentences in enumerate(text):
        if sentiment[index] == -1:
            sentimentpFinal = sentimentp[index][-1]
        elif sentiment[index] == 0:  
            sentimentpFinal = sentimentp[index][0]
        else:  
            sentimentpFinal = sentimentp[index][1]  
            
        sentimentpFinal3 = "{}%".format(round(sentimentpFinal*100,3))
        finaldata.append((sentences, sentiment[index], sentimentpFinal3))
           
    # Convert the list into a Pandas DataFrame.
    dfmessages = pd.DataFrame(finaldata, columns = ['sentences','sentiment', 'Probability']) 
    
    # append new review and sentiment to existing dataframe
    dfmessages2 = dfmessages[['sentences','sentiment']].replace(["negative","neutral","positive"],[-1,0,1] )
    dfmessages2.to_csv('Restaurant_Reviews_pos_neu_neg.csv', mode='a', index=False, header=False)
    
    dfmessages = dfmessages.replace([-1,0,1], ["negative","neutral","positive"])      
    return dfmessages    

    
def get_jvnr(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_process(preprocessedtext):
    # Create lemmatizer
    lem = WordNetLemmatizer()   
    finalprocessedtext = []
    for word in preprocessedtext:
        text_pos = pos_tag(word_tokenize(word))
        words = [x[0] for x in text_pos]
        pos = [x[1] for x in text_pos]
        word_stm = " ".join([lem.lemmatize(a,get_jvnr(b)) for a,b in zip(words,pos)])
        finalprocessedtext.append(word_stm)
    return finalprocessedtext
    
def stemming_process(preprocessedtext):
    # Create stemming
    stm = PorterStemmer()   
    finalprocessedtext = []
    for word in preprocessedtext:
        text_pos = pos_tag(word_tokenize(word))
        words = [x[0] for x in text_pos]
        pos = [x[1] for x in text_pos]
        word_stm = " ".join([stm.stem(a,get_jvnr(b)) for a,b in zip(words,pos)])
        finalprocessedtext.append(word_stm)
    return finalprocessedtext    

def preprocess(preprocessedtext):     
    processedText = []
    for Review_data in preprocessedtext:
        Review_data = str(Review_data).lower()
        Review_data = re.sub(r'#[A-Za-z0–9]+', '', Review_data) #remove hashtags
        Review_data=re.sub(r'@[A-Za-z0–9]+', '',Review_data) #remove usernames    
        Review_data=re.sub(r'@\w+', ' ', Review_data) #remove usernames
        Review_data= re.sub(r'\b\w{1}\b', '', Review_data) #remove stopwords   
        Review_data = re.sub(r'&(?![A-Za-z]+[0-9]*;|#[0-9]+;|#x[0-9a-fA-F]+;)', '', Review_data)
        Review_data = re.sub(r'&amp', '', Review_data) 
        Review_data = re.sub('\n', '', Review_data) #Remove line breaks.
        Review_data = re.sub('[%s]' % re.escape(string.punctuation), '', Review_data) #remove punctuation
        Review_data = re.sub('\[.*?\]', '', Review_data)
        Review_data=re.sub(r'http\S+', ' ', Review_data) #remove all Url
        Review_data = re.sub(r'https?:\/\/.*[\r\n]*', '', Review_data) #remove website
        Review_data = re.sub('https?://\S+|www\.\S+', '', Review_data)  #remove all websites 
        Review_data = re.sub(r' +', ' ', Review_data) #remove extra space
        Review_data = re.sub('<.*?>+', '', Review_data)    
        Review_data = re.sub('\w*\d\w*', '', Review_data)
        Review_data = re.sub(r'^RT[\s]+', '', Review_data)    
        Review_data = re.sub(r'[^a-z A-Z]', ' ',Review_data) #Remove all not characters        
        processedText.append(Review_data)        
    return processedText

#to show  progress bar
def show_progress():
    the_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        the_bar.progress(percent_complete + 1)
        
#Create API
st.title('Sentiment Analysis Application Tool')
st.write('Sentiment Analysis Review Prediction')
st.sidebar.subheader("Enter texts here, separated by semicolon")
texts = st.sidebar.text_area("Examples", value="The food is good; The juice is too sour; I feel disappointed", height=70, max_chars=None, key=None)
cols = ["texts"]
   
if (st.sidebar.button('Predict Sentiment')):   #to create progress bar
    show_progress() #to show  progress bar
    
    vectorizer, model = load('vectorizer.pickle', 'sentimentanalysis_LR.pickle')                              
    result_df = predict(vectorizer, model, texts, cols)
    st.table(result_df)
    st.text("")
    st.text("")
    st.text("")    
