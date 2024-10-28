import re
import unicodedata
import string
import nltk
import spacy
import os.path
import codecs

nlp = spacy.load("fr_core_news_sm")

def preprocess(text, lemma=False):
    """
    Transforms text to remove unwanted bits.
    """
    
    # Characters suppression 
    
    # Non normalized char suppression 
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    
    # Lowercase transformation
    text = text.lower()

    # Punctuation suppression
    translation_table = str.maketrans("", "", string.punctuation + '\n\r\t')
    text = text.translate(translation_table)
    
    # Digits suppression
    text = re.sub(r'\d', '', text)
    
    # Use the sub function to replace double spaces with single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # HTML tags removal
    text = re.sub("<.*?>", "", text)
 
    if lemma:
        lemmatised_text = nlp(text)
        text = [str(word.lemma_) for word in lemmatised_text]
        text = ' '.join(text)
    
    return text


# Chargement des données:
def load_pres(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs
